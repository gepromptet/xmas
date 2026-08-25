# Broker architecture adapted from CANTANTE by Tom Zehle:
# https://github.com/finitearth/cantante
"""The broker that lets per-agent optimizers share one system evaluation.

The problem it solves is a shape mismatch. A promptolution optimizer improves one
prompt and calls ``task.evaluate`` whenever it wants to know how a candidate did.
A multi-agent system cannot answer that question for one agent alone: the only
thing measurable is the whole system, on a *joint* prompt set. Running one system
evaluation per agent per candidate would multiply the cost by the number of
agents and would still measure the wrong thing, because the other agents' prompts
would be arbitrary.

So the optimizers are suspended instead. Each runs in its own thread against a
proxy task that blocks on ``evaluate`` and registers what it wanted scored. The
broker collects those requests, combines them into joint prompt sets, evaluates
the system once, attributes the result per agent, and wakes each optimizer with
its own share. Every optimizer sees an ordinary task and needs no modification,
which is the whole point of the seam.

The threads are a coroutine emulation, not concurrency for speed. They exist
because ``BaseOptimizer.optimize`` is a blocking call with no way to ask it what
it would like evaluated next.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from promptolution.tasks.base_task import BaseTask, EvalResult

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.prompt import Prompt

    from xmas.tasks.base import MASTask


@dataclass
class Request:
    """One agent's pending ask: score these prompts for me.

    Attributes:
        agent: The agent whose prompts these are.
        prompts: The prompt texts the inner optimizer wants scored.
        done: Set once the broker has filled in ``scores`` or ``error``.
        scores: Per-prompt, per-datapoint attributed credit. Filled by the broker.
        error: Raised into the waiting optimizer instead, if the round failed.
    """

    agent: str
    prompts: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    scores: list[list[float]] | None = None
    error: BaseException | None = None


class ProxyTask(BaseTask):
    """A task that answers with attributed credit instead of running anything.

    Handed to one agent's inner optimizer in place of the real task. Every
    ``evaluate`` call parks the caller until the broker has run a joint system
    evaluation and split the credit. The optimizer cannot tell the difference,
    which is what lets an unmodified promptolution optimizer drive one node of a
    multi-agent system.

    Attributes:
        agent: The agent this proxy speaks for.
        pending: The unanswered request, or ``None``.
    """

    def __init__(self, agent: str, task: MASTask, board: Board) -> None:
        """Initialize the proxy.

        Args:
            agent: The agent whose prompts this proxy scores.
            task: The real task, whose dataset shape the proxy mirrors so that the
                inner optimizer subsamples and blocks exactly as it would have.
            board: The shared board the broker watches.
        """
        super().__init__(
            df=task.df,
            x_column=task.x_column,
            y_column=task.y_column,
            task_description=task.task_description,
            n_subsamples=task.n_subsamples,
            eval_strategy=task.eval_strategy,
            seed=task.seed,
        )
        self.task_type = "reward"
        self.agent = agent
        self.board = board
        self.pending: Request | None = None
        self._cache: dict[str, list[float]] = {}

    def evaluate(
        self,
        prompts: Prompt | list[Prompt],
        predictor: Any = None,  # noqa: ARG002
        system_prompts: Any = None,  # noqa: ARG002
        eval_strategy: Any = None,
        block_idx: Any = None,  # noqa: ARG002
    ) -> EvalResult:
        """Return attributed credit for this agent's candidate prompts.

        Args:
            prompts: The candidate prompts.
            predictor: Ignored. The broker owns the predictor, because a candidate
                for one agent is only measurable inside a whole system.
            system_prompts: Ignored, for the same reason.
            eval_strategy: ``"evaluated"`` replays cached credit without asking
                the broker again, matching promptolution's own contract.
            block_idx: Ignored; the broker decides what gets run.

        Returns:
            An ``EvalResult`` whose scores are credit, not task reward.

        Raises:
            RuntimeError: If the broker fails the round, or if two threads
                evaluate against one proxy at once.
        """
        texts = [str(prompt) for prompt in ([prompts] if not isinstance(prompts, list) else prompts)]
        if eval_strategy == "evaluated":
            return self._result(texts, [self._cache.get(text, []) for text in texts])

        request = Request(agent=self.agent, prompts=texts)
        with self.board.condition:
            if self.pending is not None:
                raise RuntimeError(f"Two concurrent evaluations on the proxy for {self.agent!r}.")
            self.pending = request
            self.board.condition.notify_all()

        request.done.wait()
        if request.error is not None:
            raise RuntimeError(f"Attribution round failed for {self.agent!r}: {request.error}") from request.error
        scores = request.scores or []
        for text, row in zip(texts, scores, strict=False):
            self._cache[text] = row
        return self._result(texts, scores)

    def _result(self, prompts: list[str], scores: list[list[float]]) -> EvalResult:
        """Wrap credit rows in the result shape promptolution's optimizers read."""
        width = max((len(row) for row in scores), default=1) or 1
        padded = [row if len(row) == width else [*row, *([np.nan] * (width - len(row)))] for row in scores]
        array = np.asarray(padded, dtype=float).reshape(len(prompts), width) if padded else np.zeros((len(prompts), 1))
        lengths = np.asarray([float(len(text.split())) for text in prompts])
        input_tokens = np.repeat(lengths[:, None], width, axis=1)
        output_tokens = np.zeros_like(input_tokens)
        return EvalResult(
            scores=array,
            agg_scores=np.nanmean(array, axis=1),
            sequences=np.full((len(prompts), width), "", dtype=object),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agg_input_tokens=np.nanmean(input_tokens, axis=1),
            agg_output_tokens=np.nanmean(output_tokens, axis=1),
        )

    def take(self) -> Request | None:
        """Remove and return the unanswered request, if there is one."""
        request, self.pending = self.pending, None
        return request

    def cancel(self) -> None:
        """Release a parked caller without an answer, during shutdown."""
        request = self.take()
        if request is not None:
            request.error = request.error or RuntimeError("The optimization round was cancelled.")
            request.done.set()

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Never called; the proxy answers from the broker, not from predictions."""
        raise NotImplementedError("A ProxyTask is answered by the broker, not by scoring predictions.")


class Board:
    """The rendezvous between parked optimizers and the broker.

    One lock, one condition. A proxy takes the lock to post a request and wakes
    the broker; the broker waits on the same condition rather than polling, so an
    idle round costs nothing and a finished round is picked up immediately.
    """

    def __init__(self) -> None:
        """Initialize an empty board."""
        self.condition = threading.Condition()
        self.proxies: dict[str, ProxyTask] = {}

    def register(self, proxy: ProxyTask) -> None:
        """Add one agent's proxy to the board."""
        self.proxies[proxy.agent] = proxy

    def collect(self) -> dict[str, Request]:
        """Take every posted request. Caller must hold the condition."""
        taken = {}
        for agent, proxy in self.proxies.items():
            request = proxy.take()
            if request is not None:
                taken[agent] = request
        return taken

    def posted(self) -> int:
        """Return how many proxies hold an unanswered request. Caller must hold the condition."""
        return sum(1 for proxy in self.proxies.values() if proxy.pending is not None)

    def cancel_all(self) -> None:
        """Release every parked caller."""
        with self.condition:
            for proxy in self.proxies.values():
                proxy.cancel()


SETTLE_TIMEOUT_S = 10.0
"""How long a posted request waits for its peers before the round serves without them.

A backstop rather than a schedule: a round normally closes the moment the last
worker arrives. This bounds the case where one of them is doing something slow, so
the round degrades to a smaller batch instead of stalling behind it.
"""

POLL_INTERVAL_S = 0.25
"""How often the broker rechecks while a round is settling."""


def _still_working(workers: list[Future[Any]], parked: int) -> int:
    """Return how many workers are neither finished nor parked on a request."""
    return sum(1 for worker in workers if not worker.done()) - parked


def rounds(
    board: Board,
    workers: list[Future[Any]],
    settle_timeout: float = SETTLE_TIMEOUT_S,
) -> Iterator[dict[str, Request]]:
    """Yield each batch of pending requests until every worker has finished.

    A round is not served the instant one agent asks. It settles first: the loop
    waits until every worker is either parked on a request of its own or finished,
    so that one system evaluation answers the whole population. Serving the first
    request to arrive would quietly spend one whole system evaluation *per agent*,
    because an inner optimizer that does work before its first ``evaluate`` (a
    meta-model call, an initial population) does not arrive at the same instant as
    its peers. That cost is the one thing this seam exists to avoid, so waiting for
    the batch is the behaviour and not an optimization.

    The loop blocks on the board's condition rather than polling it. It wakes when
    a proxy posts a request, and on a short timeout so that a worker that ends
    without asking for anything, or one that dies, is noticed rather than waited
    on forever.

    Args:
        board: The shared board.
        workers: The futures running the per-agent optimizers.
        settle_timeout: Seconds a posted request waits for its peers before the
            round is served without them.

    Yields:
        Pending requests keyed by agent, at least one per yield.

    Raises:
        BaseException: Whatever a worker raised, once it is seen to have failed.
    """
    while True:
        with board.condition:
            deadline: float | None = None
            while True:
                parked = board.posted()
                if parked:
                    if deadline is None:
                        deadline = time.monotonic() + settle_timeout
                    if _still_working(workers, parked) <= 0 or time.monotonic() >= deadline:
                        break
                else:
                    failed = next((worker for worker in workers if worker.done() and worker.exception()), None)
                    if failed is not None:
                        raise failed.exception() or RuntimeError("A worker failed without an exception.")
                    if all(worker.done() for worker in workers):
                        return
                board.condition.wait(timeout=POLL_INTERVAL_S)
            batch = board.collect()
        yield batch


def run_workers(
    executor: ThreadPoolExecutor,
    steps: dict[str, Any],
) -> list[Future[Any]]:
    """Start one worker per agent.

    Args:
        executor: The pool to run them on.
        steps: Zero-argument callables keyed by agent name.

    Returns:
        The futures, in the order the agents were given.
    """
    return [executor.submit(step) for step in steps.values()]
