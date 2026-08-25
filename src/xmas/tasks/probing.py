"""Generation-free evaluation: score candidates by probing a shared reference pool.

Evaluation dominates the cost of optimizing a multi-agent system, and it dominates
for a mechanical reason: it decodes. Producing an output of ``L`` tokens costs ``L``
sequential forward passes. Teacher-forcing an output that already exists costs one
parallel pass, because every token is known in advance.

So this module never asks a candidate what it would say. It asks how much
probability a candidate puts on outputs that were said already, and whose reward is
therefore known. A candidate that concentrates probability on references known to be
good is credited; one that concentrates it on references known to be bad is not.

The estimator is CALINKA's importance-sampling form (gepromptet/calinka,
``paper/calinka.tex``)::

    J_IS(pi) = 1/(nK) * sum_i sum_k  w_pi(y_k^i | x^i) * R(y_k^i, y^i)

    w_pi(y | x) = p(y | x, pi) / q(y | x)

    q(y | x)    = 1/M * sum_m p(y | x, pi_m)

``p̃`` is the raw teacher-forced sequence probability and ``R`` is the reference's own
reward, recorded when it was decoded and never the reward of the candidate under
test. The denominator mixes over the whole candidate population, the candidate
included; there is no leave-one-out. That population term is why
:meth:`ProbingTask.evaluate` needs every candidate at once, and it is the reason the
class overrides ``evaluate`` rather than plugging into promptolution's per-prompt
caching: a score computed against one population is not valid against another.

The seam this hangs on is the one design principle P2 names. An optimizer reaches a
task only through ``evaluate``, so wrapping any :class:`~xmas.tasks.base.MASTask` in a
:class:`ProbingTask` swaps generation for probing without a single optimizer learning
that anything changed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from promptolution.tasks.base_task import EvalResult
from promptolution.utils.prompt import Prompt

from xmas.prompts import decode_prompts, safe_format
from xmas.tasks.base import MASTask

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor

    from xmas.llms.scoring import SupportsScore
    from xmas.mas import MAS

DEFAULT_REFS_PER_INPUT = 3
"""References sampled per evaluation input when a pool is generated."""


@dataclass(frozen=True)
class Reference:
    """One output that was decoded already, with the reward it earned.

    A reference is not tied to the candidate that produced it. It is a fixed piece
    of text plus a known reward, and every candidate in the population is asked the
    same question about it: how likely were you to have said this?

    Attributes:
        x: The evaluation input this reference answers.
        inputs: The state the terminal agent read when this reference was produced,
            keyed by variable name. For a single-agent system this is just
            ``{"query": x}``; for a deeper one it carries the upstream agents'
            outputs, which is what lets a candidate's terminal prompt be rendered
            exactly as it would have been at decode time.
        completion: The terminal agent's raw response, i.e. the text to teacher-force.
        reward: The reward this completion earned against its target, recorded when
            it was decoded.
        source: Name of the candidate that authored it. Diagnostic only; the
            estimator never reads it, because the mixture runs over the whole
            population rather than over each reference's own author.
    """

    x: str
    inputs: dict[str, str]
    completion: str
    reward: float
    source: str | None = None


class ReferencePool:
    """The shared pool of references every candidate is probed against.

    Shared is the operative word: the cost saving exists because one pool serves the
    whole population, so adding a candidate costs probes rather than decodes.

    Attributes:
        references: Every reference, in insertion order.
    """

    references: tuple[Reference, ...]

    def __init__(self, references: Sequence[Reference]) -> None:
        """Initialize the pool.

        Args:
            references: The references. May be empty only if the pool is about to be
                filled by :meth:`generate`.

        Raises:
            ValueError: If a reference carries an empty completion, which cannot be
                teacher-forced.
        """
        for reference in references:
            if not reference.completion:
                raise ValueError(
                    f"The reference for input {reference.x!r} has an empty completion, which has no "
                    "sequence log-probability. Drop it from the pool rather than probing it."
                )
        self.references = tuple(references)

        self._by_input: dict[str, list[Reference]] = {}
        for reference in self.references:
            self._by_input.setdefault(reference.x, []).append(reference)

    @property
    def inputs(self) -> tuple[str, ...]:
        """The evaluation inputs this pool covers, in insertion order."""
        return tuple(self._by_input)

    def for_input(self, x: str) -> tuple[Reference, ...]:
        """Return the references for one input, in insertion order.

        Args:
            x: The evaluation input.

        Returns:
            Its references. Empty if the pool does not cover this input.
        """
        return tuple(self._by_input.get(x, ()))

    def columns(self, xs: Sequence[str], k: int | None = None) -> tuple[Reference, ...]:
        """Return the references to probe for a set of inputs.

        Every candidate is scored against the *same* columns, which is what makes the
        importance weights comparable across candidates.

        Args:
            xs: The evaluation inputs, in the order the caller wants them grouped.
            k: Reference budget per input. ``None`` uses every reference the pool
                holds. Taking the first ``k`` in insertion order rather than sampling
                keeps a budget sweep deterministic and nested, so ``k`` and ``k + 1``
                differ by exactly one column per input.

        Returns:
            The selected references, grouped by input in the order ``xs`` gives.

        Raises:
            ValueError: If ``k`` is not positive, or no input has a reference.
        """
        if k is not None and k < 1:
            raise ValueError(f"The reference budget must be at least 1, got {k}.")

        selected: list[Reference] = []
        for x in xs:
            available = self.for_input(x)
            selected.extend(available if k is None else available[:k])

        if not selected:
            raise ValueError(
                f"The reference pool covers none of the {len(xs)} inputs being evaluated. A pool is "
                "built for a specific set of inputs; build it from the same subsample the task evaluates."
            )
        return tuple(selected)

    @classmethod
    def generate(
        cls,
        task: MASTask,
        candidates: Sequence[MAS],
        xs: Sequence[str],
        ys: Sequence[str],
        refs_per_input: int = DEFAULT_REFS_PER_INPUT,
        seed: int = 42,
    ) -> ReferencePool:
        """Decode a pool by running candidates drawn uniformly from the population.

        The source of each reference is drawn i.i.d. uniformly over *all* candidates,
        because that is the proposal distribution the estimator's denominator divides
        by. Drawing them any other way, favouring good candidates for instance, makes
        the mixture ``q̃`` describe something other than where the references came
        from, and the importance weights then correct towards the wrong target.

        This is the one step that decodes. It happens once per pool and is then
        amortized over every candidate and every optimizer step that reuses it.

        Args:
            task: The task whose reward grades the references.
            candidates: The candidate population to draw sources from.
            xs: The evaluation inputs.
            ys: The targets, aligned to ``xs``.
            refs_per_input: References to decode per input.
            seed: Seed for the source draw, so a pool is reproducible.

        Returns:
            The decoded pool.

        Raises:
            ValueError: If no candidates are given, or ``refs_per_input`` is not
                positive.
        """
        if not candidates:
            raise ValueError("A reference pool needs at least one candidate to decode from.")
        if refs_per_input < 1:
            raise ValueError(f"refs_per_input must be at least 1, got {refs_per_input}.")

        rng = np.random.default_rng(seed)
        sources = rng.integers(0, len(candidates), size=(refs_per_input, len(xs)))

        references: list[Reference] = []
        for slot in range(refs_per_input):
            for index in np.unique(sources[slot]):
                candidate = candidates[int(index)]
                positions = [i for i in range(len(xs)) if sources[slot][i] == index]
                traces = candidate.run_batch([xs[i] for i in positions])
                terminal = candidate.terminal
                completions = [_terminal_output(trace, terminal) for trace in traces]
                rewards = task.reward(
                    [xs[i] for i in positions],
                    [ys[i] for i in positions],
                    [completion for completion, _ in completions],
                )
                for position, (completion, inputs), score in zip(positions, completions, rewards, strict=True):
                    if not completion:
                        continue
                    references.append(
                        Reference(
                            x=xs[position],
                            inputs=inputs,
                            completion=completion,
                            reward=float(score),
                            source=_candidate_name(candidate),
                        )
                    )

        return cls(references)

    def __len__(self) -> int:
        """Number of references in the pool."""
        return len(self.references)

    def __repr__(self) -> str:
        """Return an unambiguous representation naming the pool's shape."""
        return f"ReferencePool(references={len(self.references)}, inputs={len(self._by_input)})"


@dataclass(frozen=True)
class ProbeTensor:
    """Teacher-forced log-probabilities of every reference under every candidate.

    This is the object an estimator computes on, and the only thing probing produces
    that costs model time. Everything downstream of it is arithmetic.

    Attributes:
        log_p: Shape ``(M, R)``. ``log_p[m, r]`` is ``log p̃`` of reference ``r``'s
            completion under candidate ``m``. ``NaN`` marks a cell that was not
            probed; it drops out of the mixture rather than counting as zero
            probability.
        reward: Shape ``(R,)``. Each reference's own recorded reward.
        input_index: Shape ``(R,)``. Which input each reference column belongs to,
            as an index into :attr:`inputs`.
        inputs: The evaluation inputs, in column-group order.
        probed_tokens: Shape ``(R,)``. Tokens scored per reference column, for the
            cost accounting. Decode tokens are zero by construction.
    """

    log_p: np.ndarray
    reward: np.ndarray
    input_index: np.ndarray
    inputs: tuple[str, ...]
    probed_tokens: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_candidates(self) -> int:
        """Number of candidates, ``M``."""
        return int(self.log_p.shape[0])

    @property
    def n_references(self) -> int:
        """Number of reference columns, ``R``."""
        return int(self.log_p.shape[1])

    def by_input(self) -> list[np.ndarray]:
        """Return the column indices belonging to each input, in :attr:`inputs` order."""
        return [np.flatnonzero(self.input_index == index) for index in range(len(self.inputs))]


def importance_weights(log_p: np.ndarray) -> np.ndarray:
    """Return ``w[m, r] = p̃(ŷ_r | π_m) / ((1/M) Σ_m' p̃(ŷ_r | π_m'))``.

    Computed through log-sum-exp and never by exponentiating ``log_p`` directly. A
    sequence log-probability of a few hundred negative nats underflows to exactly
    zero in float64, which would turn every weight into ``0/0``; the ratio itself is
    perfectly well conditioned, so it is the ratio that gets evaluated.

    Cells that were not probed stay ``NaN`` and are excluded from their column's
    mixture, which is then normalized by the number of candidates actually present.

    Args:
        log_p: Shape ``(M, R)`` teacher-forced sequence log-probabilities.

    Returns:
        Shape ``(M, R)`` importance weights, ``NaN`` wherever ``log_p`` was.
    """
    present = ~np.isnan(log_p)
    counts = present.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        shifted = np.where(present, log_p, -np.inf)
        peak = np.max(shifted, axis=0)
        exponentiated = np.where(present, np.exp(log_p - peak), 0.0)
        log_q = peak + np.log(exponentiated.sum(axis=0)) - np.log(counts)
        return np.where(present, np.exp(log_p - log_q), np.nan)


def _row_means(values: np.ndarray) -> np.ndarray:
    """Return each row's mean over its non-``NaN`` entries, ``NaN`` if it has none.

    ``np.nanmean`` does this but warns on an all-``NaN`` row, and an all-``NaN`` row
    is a legitimate outcome here: it is a candidate nothing could be measured on.
    Warning about the case the caller is deliberately representing would train the
    reader to ignore warnings.

    Args:
        values: Shape ``(rows, columns)``.

    Returns:
        Shape ``(rows,)`` means.
    """
    present = ~np.isnan(values)
    counts = present.sum(axis=1)
    totals = np.where(present, values, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


class Estimator(Protocol):
    """Turns a probe tensor into a score per candidate per input.

    A new estimator is a new component rather than a new framework (design principle
    P4): implement this one method and every optimizer keeps working.
    """

    def __call__(self, probe: ProbeTensor) -> np.ndarray:
        """Score every candidate on every input.

        Args:
            probe: The teacher-forced log-probabilities and reference rewards.

        Returns:
            Shape ``(M, n_inputs)`` scores.
        """
        ...


class ImportanceSampling:
    """CALINKA's estimator: reward-weighted, population-normalized probe mass.

    Per input, a candidate's score is the mean over that input's references of the
    reference's reward times the candidate's importance weight on it. The population
    mixture in the denominator is what removes the reference's own difficulty: a long
    or fluent reference is easy for *every* candidate, so dividing by what the
    population as a whole assigns to it cancels the part that says nothing about
    which candidate is better.

    The absolute value is not an unbiased estimate of expected reward and is not
    meant to be read as one. What it is meant to preserve is the ordering, because
    that is all a selection step consumes.
    """

    def __call__(self, probe: ProbeTensor) -> np.ndarray:
        """Score every candidate on every input.

        Args:
            probe: The probe tensor.

        Returns:
            Shape ``(M, n_inputs)`` scores. A cell is ``NaN`` when none of that
            input's references could be probed, which is missing measurement rather
            than a score of zero.
        """
        weights = importance_weights(probe.log_p)
        contribution = weights * probe.reward[None, :]

        columns = probe.by_input()
        scores = np.full((probe.n_candidates, len(columns)), np.nan)
        for index, group in enumerate(columns):
            if group.size == 0:
                continue
            scores[:, index] = _row_means(contribution[:, group])
        return scores

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return "ImportanceSampling()"


class ProbingTask(MASTask):
    """Evaluates candidates by probing a reference pool instead of generating.

    Wraps any :class:`~xmas.tasks.base.MASTask` and presents the same ``evaluate``
    contract, so an optimizer written against the wrapped task runs against this one
    unchanged. That is design principle P2 made concrete: evaluation strategy is a
    property of the task, invisible to whoever is optimizing.

    What it probes is the **terminal agent**, whose output is the system's output.
    For a single-agent system that is the whole system, and the estimator is exactly
    the single-prompt method. For a deeper one, each reference carries the state its
    terminal agent actually read at decode time, so a candidate's terminal prompt is
    rendered against real upstream context rather than against unfilled placeholders.
    The upstream agents are held fixed at the pool's values; probing credits the
    terminal component only, and a prompt change further upstream is invisible here.

    Attributes:
        task: The wrapped task, whose reward and dataset this one reuses.
        references: The shared pool every candidate is probed against.
        estimator: Turns the probe tensor into scores.
        k: Reference budget per input, or ``None`` for the whole pool.
    """

    def __init__(
        self,
        task: MASTask,
        references: ReferencePool,
        estimator: Estimator | None = None,
        k: int | None = None,
        llm: SupportsScore | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the probing task.

        Args:
            task: The task to wrap. Its dataset, columns and reward are reused.
            references: The pool to probe against.
            estimator: How probes become scores. Defaults to
                :class:`ImportanceSampling`.
            k: Reference budget per input. ``None`` probes the whole pool.
            llm: Model to probe with. Defaults to whichever model the candidate's
                terminal agent is bound to, so a per-node binding is respected and
                the probe runs on the model that would have generated.
            **kwargs: Forwarded to :class:`~xmas.tasks.base.MASTask`.

        Raises:
            ValueError: If the pool is empty.
        """
        if not len(references):
            raise ValueError("A ProbingTask needs a non-empty reference pool.")

        super().__init__(
            df=task.df,
            x_column=task.x_column,
            y_column=task.y_column or task.x_column,
            task_description=task.task_description,
            n_subsamples=task.n_subsamples,
            eval_strategy=task.eval_strategy,
            seed=task.seed,
            **kwargs,
        )
        self.task = task
        self.references = references
        self.estimator = estimator if estimator is not None else ImportanceSampling()
        self.k = k
        self.llm = llm
        self._probe_cache: dict[tuple[str, str], float] = {}
        self._token_cache: dict[tuple[str, str], int] = {}

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Delegate the reward to the wrapped task.

        Reached only if something calls the generation path directly; probing itself
        goes through :meth:`evaluate`. It delegates rather than raising so that the
        wrapped task's reward stays the one definition of what a good answer is.

        Args:
            xs: The inputs.
            ys: The targets.
            preds: The outputs to score.

        Returns:
            One score per datapoint.
        """
        return self.task.reward(xs, ys, preds)

    def evaluate(
        self,
        prompts: Prompt | list[Prompt],
        predictor: BasePredictor,
        system_prompts: str | list[str] | None = None,  # noqa: ARG002
        eval_strategy: Any = None,
        block_idx: int | list[int] | None = None,
    ) -> EvalResult:
        """Score a whole candidate population by probing, not by generating.

        The population is evaluated together because the estimator is
        population-relative: a candidate's weight on a reference is its share of what
        every candidate assigns to that reference. Scoring one candidate alone is not
        a smaller version of this computation, it is a different one, which is also
        why promptolution's per-prompt evaluation cache is bypassed. The cache that
        does apply sits one level down, on ``(context, completion)`` probe cells,
        which are population-independent and dominate the cost.

        Args:
            prompts: Encoded prompt sets, one per candidate.
            predictor: The MAS the prompt sets belong to. Supplies the topology, the
                terminal agent and the model binding; its own prompts are replaced by
                each candidate's.
            system_prompts: Accepted for interface compatibility, not applied.
            eval_strategy: Subsampling strategy, as on the wrapped task.
            block_idx: Specific block indices to evaluate.

        Returns:
            One score per candidate per input, aggregated per candidate. Output
            tokens are zero throughout: probing decodes nothing.
        """
        candidates = [prompts] if isinstance(prompts, Prompt) else list(prompts)
        if isinstance(block_idx, int):
            block_idx = [block_idx]
        xs, _ = self.subsample(eval_strategy=eval_strategy or self.eval_strategy, block_idx=block_idx)

        probe = self.probe(candidates, predictor, xs)
        scores = self.estimator(probe)
        agg_scores = _row_means(scores) if scores.size else np.array([])

        input_tokens = np.zeros_like(scores)
        for index, group in enumerate(probe.by_input()):
            if group.size:
                input_tokens[:, index] = probe.probed_tokens[group].sum()
        output_tokens = np.zeros_like(scores)

        return EvalResult(
            scores=scores,
            agg_scores=agg_scores,
            sequences=np.full(scores.shape, "", dtype=object),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agg_input_tokens=input_tokens.mean(axis=1) if input_tokens.size else np.array([]),
            agg_output_tokens=output_tokens.mean(axis=1) if output_tokens.size else np.array([]),
        )

    def probe(self, candidates: Sequence[Prompt], predictor: BasePredictor, xs: Sequence[str]) -> ProbeTensor:
        """Teacher-force every selected reference under every candidate.

        Args:
            candidates: Encoded prompt sets, one per candidate.
            predictor: The MAS the prompt sets belong to.
            xs: The evaluation inputs to probe over.

        Returns:
            The probe tensor.

        Raises:
            TypeError: If the resolved model cannot score a continuation.
        """
        mas = _as_mas(predictor)
        terminal = mas.terminal
        scorer = self.llm if self.llm is not None else _resolve_scorer(mas, terminal)

        columns = self.references.columns(list(xs), k=self.k)
        inputs = tuple(dict.fromkeys(reference.x for reference in columns))
        position = {x: index for index, x in enumerate(inputs)}

        templates = [decode_prompts(str(prompt), list(mas.agents))[terminal] for prompt in candidates]

        log_p = np.full((len(templates), len(columns)), np.nan)
        probed_tokens = np.zeros(len(columns))
        for row, template in enumerate(templates):
            for column, reference in enumerate(columns):
                context = safe_format(template, reference.inputs)
                total, tokens = self._probe_cell(scorer, context, reference.completion)
                log_p[row, column] = total
                probed_tokens[column] = tokens

        return ProbeTensor(
            log_p=log_p,
            reward=np.array([reference.reward for reference in columns], dtype=float),
            input_index=np.array([position[reference.x] for reference in columns], dtype=int),
            inputs=inputs,
            probed_tokens=probed_tokens,
        )

    def _probe_cell(self, scorer: SupportsScore, context: str, completion: str) -> tuple[float, int]:
        """Return one cell's sequence log-probability and token count, memoized.

        Two references that decoded to the same text under the same rendered context
        have the same log-probability by definition, so the pair is the cache key and
        repeats cost nothing. The cache is keyed on content rather than on identity,
        which also makes it survive a pool being rebuilt.

        Args:
            scorer: The model to probe with.
            context: The rendered terminal prompt.
            completion: The reference text to teacher-force.

        Returns:
            The summed log-probability and the number of tokens it covered.
        """
        key = (context, completion)
        if key not in self._probe_cache:
            scored = scorer.score(context, completion)
            self._probe_cache[key] = scored.sum_logprob
            self._token_cache[key] = scored.token_count
        return self._probe_cache[key], self._token_cache[key]

    def __repr__(self) -> str:
        """Return an unambiguous representation naming the wrapped task and pool."""
        return f"ProbingTask(task={type(self.task).__name__}, references={len(self.references)}, k={self.k})"


def _terminal_output(trace: Any, terminal: str) -> tuple[str, dict[str, str]]:
    """Return a trace's terminal completion and the state its terminal agent read."""
    records = trace[terminal]
    if not records:
        return "", {}
    record = records[-1]
    return str(record.raw_response), dict(record.inputs)


def _candidate_name(candidate: MAS) -> str:
    """Return a stable name for a candidate, used only as reference provenance."""
    return candidate.encode()


def _as_mas(predictor: BasePredictor) -> MAS:
    """Narrow a predictor to a MAS, which is what probing needs to render a prompt."""
    from xmas.mas import MAS as _MAS

    if not isinstance(predictor, _MAS):
        raise TypeError(
            f"ProbingTask needs a MAS predictor to render a terminal prompt, got {type(predictor).__name__}."
        )
    return predictor


def _resolve_scorer(mas: MAS, terminal: str) -> SupportsScore:
    """Return the model serving the terminal agent, if it can score.

    Raises:
        TypeError: If that model has no ``score``, naming it so the caller knows
            which binding to change rather than which line failed.
    """
    llm = mas.resolve_llm(mas.agents[terminal])
    scorer: Callable[..., Any] | None = getattr(llm, "score", None)
    if not callable(scorer):
        raise TypeError(
            f"Probing needs a model that can teacher-force a continuation, but the model bound to "
            f"agent {terminal!r} ({type(llm).__name__}) has no score(). Pass llm=... explicitly, or "
            "bind that agent to a scoring-capable backend."
        )
    return llm  # type: ignore[return-value]


def rank_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman rank correlation between two candidate scorings.

    Written here rather than pulled from scipy because it is six lines and scipy is
    not a dependency of a package whose point is a small install.

    Args:
        left: Scores per candidate under one evaluation strategy.
        right: Scores per candidate under another, aligned to ``left``.

    Returns:
        Spearman's rho in ``[-1, 1]``, or ``nan`` when either side has no variance
        and a correlation is therefore undefined.

    Raises:
        ValueError: If the two scorings have different lengths.
    """
    if len(left) != len(right):
        raise ValueError(f"Cannot correlate {len(left)} scores against {len(right)}.")
    if len(left) < 2:
        return math.nan

    first, second = _ranks(left), _ranks(right)
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = math.sqrt(float((first_centered**2).sum()) * float((second_centered**2).sum()))
    if denominator == 0.0:
        return math.nan
    return float((first_centered * second_centered).sum() / denominator)


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Return average ranks, so ties do not fabricate an ordering."""
    array = np.asarray(values, dtype=float)
    order = array.argsort()
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)

    for value in np.unique(array):
        tied = np.flatnonzero(array == value)
        if tied.size > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


@dataclass(frozen=True)
class AgreementReadout:
    """How closely probing reproduces the ranking that generation would give.

    The diagnostic the whole method rests on. Probing is only a legitimate stand-in
    for generation if it puts the candidates in the same order, because an
    optimizer's selection step consumes the order and nothing else. The absolute
    values are not comparable and are not meant to be: one is a mean reward, the
    other a reward-weighted probability share.

    Attributes:
        generation: Ĵ_AR per candidate, the mean reward of what each actually decoded.
        probing: Ĵ_IS per candidate, from the probe tensor.
        rho: Spearman rank correlation between the two, over the candidates.
        generation_tokens: Mean tokens decoded per datapoint, summed over candidates.
        probing_tokens: Mean tokens teacher-forced per datapoint, summed over
            candidates. Decode tokens are zero, which is the point.
        null_rho_mean: Mean rank correlation once the probe tensor is shuffled.
        null_rho_ci: Central 95% interval of those shuffled correlations.
    """

    generation: np.ndarray
    probing: np.ndarray
    rho: float
    generation_tokens: float
    probing_tokens: float
    null_rho_mean: float = math.nan
    null_rho_ci: tuple[float, float] = (math.nan, math.nan)

    @property
    def admissible(self) -> bool:
        """Whether the negative control passed, i.e. its interval contains zero.

        Shuffling the probe tensor destroys which candidate assigned what, while
        leaving the reference rewards and the marginal spread of log-probabilities
        untouched. If the correlation survives that, it was coming from the rewards
        or from the shape of the population rather than from the probes, and
        :attr:`rho` is not evidence about the estimator. A readout that is not
        admissible should be reported as a failed guard, not as a weak result.
        """
        low, high = self.null_rho_ci
        return bool(not math.isnan(low) and low <= 0.0 <= high)

    @property
    def speedup(self) -> float:
        """Decoded tokens per probed token, or ``inf`` if nothing was probed.

        A token-count ratio only. The wall-clock advantage is larger, because
        decoding is sequential and a probe is one parallel pass, but that factor is
        a property of the serving stack and has to be measured there rather than
        inferred here.
        """
        if self.probing_tokens == 0.0:
            return math.inf
        return self.generation_tokens / self.probing_tokens

    def __repr__(self) -> str:
        """Return the readout's headline numbers."""
        return (
            f"AgreementReadout(rho={self.rho:.3f}, null={self.null_rho_mean:.3f}, "
            f"admissible={self.admissible}, candidates={len(self.generation)}, "
            f"token_ratio={self.speedup:.1f}x)"
        )


def negative_control(
    probe: ProbeTensor,
    target: Sequence[float],
    estimator: Estimator | None = None,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Return the mean and 95% interval of the rank correlation under a shuffled probe.

    The guard calinka runs before it trusts any recovery curve. Permuting the whole
    tensor breaks the association between a candidate and what it assigned, but keeps
    the reference rewards and the marginal distribution of log-probabilities exactly
    as they were, so anything that still correlates was never coming from the probes.

    Args:
        probe: The measured probe tensor.
        target: The generation-based score per candidate.
        estimator: The estimator to re-run. Defaults to :class:`ImportanceSampling`.
        n_permutations: Permutations to draw.
        seed: Seed, so a guard is reproducible.

    Returns:
        The mean null correlation and its central 95% interval.
    """
    estimate = estimator if estimator is not None else ImportanceSampling()
    rng = np.random.default_rng(seed)
    flat = probe.log_p.flatten()

    draws: list[float] = []
    for _ in range(n_permutations):
        shuffled = _with_log_p(probe, rng.permutation(flat).reshape(probe.log_p.shape))
        rho = rank_agreement(list(target), _row_means(estimate(shuffled)).tolist())
        if not math.isnan(rho):
            draws.append(rho)

    if not draws:
        return math.nan, (math.nan, math.nan)
    values = np.array(draws)
    return float(values.mean()), (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def _with_log_p(probe: ProbeTensor, log_p: np.ndarray) -> ProbeTensor:
    """Return a copy of a probe tensor carrying different log-probabilities."""
    return ProbeTensor(
        log_p=log_p,
        reward=probe.reward,
        input_index=probe.input_index,
        inputs=probe.inputs,
        probed_tokens=probe.probed_tokens,
    )


def materialize(predictor: MAS, candidates: Sequence[Prompt]) -> list[MAS]:
    """Turn encoded prompt sets back into the systems they describe.

    A candidate travels through promptolution as a string, but decoding a reference
    pool means actually running each one, and the estimator's denominator assumes the
    pool's sources were drawn uniformly over exactly this population. Handing the pool
    builder a single system standing in for all of them would make the mixture that
    ``q̃`` averages describe a proposal the references did not come from, and the
    weights would then correct towards the wrong target.

    Args:
        predictor: The system the prompt sets belong to; supplies topology and models.
        candidates: Encoded prompt sets.

    Returns:
        One system per candidate, sharing ``predictor``'s structure and models.
    """
    names = list(predictor.agents)
    return [predictor.copy(prompts=decode_prompts(str(prompt), names)) for prompt in candidates]


def measure_agreement(
    task: MASTask,
    predictor: MAS,
    candidates: Sequence[Prompt],
    references: ReferencePool | None = None,
    k: int | None = None,
    refs_per_input: int = DEFAULT_REFS_PER_INPUT,
    seed: int = 42,
) -> AgreementReadout:
    """Score one candidate set both ways and report how far the rankings agree.

    Generation and probing are measured on the same inputs and the same candidates,
    so the only thing that differs is the estimator. The reference pool is decoded
    separately from the generation-based evaluation rather than reusing its outputs:
    a pool built from the very decodes being used as the target would let each
    candidate probe its own answers, and the agreement would then partly measure that
    overlap rather than the estimator.

    Args:
        task: The task to evaluate against.
        predictor: The MAS the candidate prompt sets belong to.
        candidates: Encoded prompt sets, one per candidate.
        references: A pool to reuse. ``None`` decodes a fresh one, drawing its
            sources uniformly over the whole candidate set.
        k: Reference budget per input. ``None`` uses the whole pool.
        refs_per_input: References to decode per input when building a pool.
        seed: Seed for the pool's source draw.

    Returns:
        The readout, including the shuffled-probe negative control. Check
        ``admissible`` before reading ``rho``.

    Raises:
        ValueError: If fewer than two candidates are given, since a rank correlation
            over one candidate is undefined.
    """
    if len(candidates) < 2:
        raise ValueError(f"An agreement readout needs at least two candidates, got {len(candidates)}.")

    generated = task.evaluate(list(candidates), predictor=predictor)

    xs, ys = task.subsample()
    pool = references or ReferencePool.generate(
        task,
        materialize(predictor, candidates),
        xs,
        ys,
        refs_per_input=refs_per_input,
        seed=seed,
    )
    probing = ProbingTask(task, pool, k=k)
    tensor = probing.probe(list(candidates), predictor, xs)
    probed = _row_means(probing.estimator(tensor))

    target = generated.agg_scores.tolist()
    null_mean, null_ci = negative_control(tensor, target, probing.estimator, seed=seed)

    return AgreementReadout(
        generation=generated.agg_scores,
        probing=probed,
        rho=rank_agreement(target, probed.tolist()),
        generation_tokens=float(np.nansum(generated.agg_output_tokens)),
        probing_tokens=float(tensor.probed_tokens.sum()) * len(candidates),
        null_rho_mean=null_mean,
        null_rho_ci=null_ci,
    )
