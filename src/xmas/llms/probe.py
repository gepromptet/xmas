"""Teacher forcing through the gateway, at a concurrency the key allows.

xmas's `APILLM.score()` is one HTTP call for one (context, continuation) pair, so
throughput is the caller's business. This is that caller.

Three properties, in the order they matter:

  * ORDER. Results come back aligned to the pairs that were asked for. A probe
    attributed to the wrong reference is not a slow run, it is a wrong tensor, and
    nothing downstream could see it.
  * THE WALL IS RESPECTED. A 429 is the system working. The endpoint says how long to
    wait and that wait is taken, once, on the call that was refused; the limiter is
    not widened and the call is not hammered. A refusal with no Retry-After falls back
    to a bounded exponential backoff rather than to a guess dressed as a number.
  * FAILURES ARE DATA. A pair that cannot be scored after its retries comes back as
    None and becomes a NaN cell in the tensor, which `estimators.importance_weights` already
    drops out of its column. Losing a cell must never lose the run.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from xmas.errors import XmasError

if TYPE_CHECKING:  # pragma: no cover
    from xmas.llms.api_llm import APILLM
    from xmas.llms.scoring import ScoredSequence

DEFAULT_CONCURRENCY = 32
"""In-flight probes. Sized to the run key; ask before raising it."""

MAX_ATTEMPTS = 4
"""Attempts per probe, including the first."""

BACKOFF_BASE_S = 2.0
"""First backoff when a refusal carries no Retry-After. Doubles per attempt."""

MAX_BACKOFF_S = 60.0
"""Ceiling on a single wait, so a stuck endpoint cannot park the run for an hour."""


class Limiter:
    """A counting semaphore with a shared cool-off.

    The semaphore caps how many probes are in flight. The cool-off is the part a plain
    semaphore does not give: when the endpoint refuses one call, every other worker
    should stop pushing too, because the limit is on the key and not on the call. So a
    429 parks all of them until the deadline the endpoint named, rather than letting
    the remaining 31 workers keep arriving at a door that just closed.
    """

    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY) -> None:
        """Initialize with a cap on concurrently in-flight probes."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        self.concurrency = concurrency
        self._slots = threading.Semaphore(concurrency)
        self._lock = threading.Lock()
        self._open_at = 0.0
        self.waits = 0
        self.waited_s = 0.0

    def hold_off(self, seconds: float) -> None:
        """Close the door for everyone until `seconds` from now.

        Extends an existing cool-off, never shortens it: two refusals arriving together
        should leave the longer wait standing.
        """
        with self._lock:
            self._open_at = max(self._open_at, time.monotonic() + max(0.0, seconds))
            self.waits += 1

    def _sleep_until_open(self) -> None:
        while True:
            with self._lock:
                remaining = self._open_at - time.monotonic()
            if remaining <= 0:
                return
            self.waited_s += remaining
            time.sleep(min(remaining, 1.0))

    def __enter__(self) -> Limiter:
        """Wait out any cool-off, then take a slot."""
        self._sleep_until_open()
        self._slots.acquire()
        # Re-check: the door may have closed while this worker queued for a slot.
        self._sleep_until_open()
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the slot. Exceptions propagate."""
        self._slots.release()


def _wait_for(error: Exception, attempt: int) -> float:
    """How long to wait after one refusal.

    The endpoint's own Retry-After when it sent one, since it knows and we do not.
    Otherwise exponential backoff from BACKOFF_BASE_S, capped. No jitter: the workers
    are already serialized behind one shared cool-off, so they do not need to be
    spread out, and jitter would only make the wait harder to account for.
    """
    asked = error.retry_after if isinstance(error, XmasError) else None
    if asked is not None:
        return min(float(asked), MAX_BACKOFF_S)
    return float(min(BACKOFF_BASE_S * (2 ** (attempt - 1)), MAX_BACKOFF_S))


def probe_all(
    scorer: APILLM,
    pairs: Sequence[tuple[str, str]],
    limiter: Limiter | None = None,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[ScoredSequence | None]:
    """Score every (context, continuation) pair, concurrently, in order.

    Args:
        scorer: An APILLM, or a subclass such as VLLMServerLLM.
        pairs: (context, continuation) tuples.
        limiter: Shared Limiter. One per run, not one per call, or the concurrency cap
            would apply per batch instead of to the key.
        on_error: Called with (index, exception) for a pair that exhausted its
            attempts. Diagnostics only; the run continues.

    Returns:
        One ScoredSequence per pair, aligned to `pairs`, or None where scoring failed.
    """
    pairs = list(pairs)
    if not pairs:
        return []
    limiter = limiter if limiter is not None else Limiter()
    results: list[ScoredSequence | None] = [None] * len(pairs)

    def one(index: int) -> None:
        context, continuation = pairs[index]
        last = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with limiter:
                    results[index] = scorer.score(context, continuation)
                return
            except Exception as error:
                last = error
                if not _retryable(error) or attempt == MAX_ATTEMPTS:
                    break
                limiter.hold_off(_wait_for(error, attempt))
        if on_error is not None and last is not None:
            on_error(index, last)

    with ThreadPoolExecutor(max_workers=limiter.concurrency, thread_name_prefix="xmas-probe") as pool:
        list(pool.map(one, range(len(pairs))))
    return results


def _retryable(error: Exception) -> bool:
    """Whether waiting could plausibly change the answer.

    A rate limit, yes: that is what Retry-After is for. A transport hiccup, yes. A
    capability refusal (the gateway's 501 for a backbone that cannot teacher-force) or
    a malformed request, no — those return the same answer however long you wait, and
    retrying them is how a run turns one configuration error into a thousand calls.
    """
    if isinstance(error, XmasError):
        return error.status == 429
    # xmas raises ValueError for a rejected request AND for a continuation that
    # does not align with token boundaries. Neither is fixed by waiting.
    return not isinstance(error, ValueError)
