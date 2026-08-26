"""Call-level statistics for the LLM layer.

An experiment whose claim is a speedup has to be able to say where its time and its
tokens went, and the only place that knows is the layer that makes the calls. So the
record is taken here rather than reconstructed afterwards from a log.

Deliberately small: a frozen record per call, a collector that appends them, and
aggregation computed on demand. No dashboard, no background thread, no persistence
beyond a JSONL dump the caller asks for.

Two honesty notes run through the whole module.

promptolution counts tokens by splitting on whitespace
(``BaseLLM.update_token_count``), so the token counts on a generation call are an
approximation, and a systematic undercount at that: English runs about 1.3 subword
tokens per whitespace word. They are labelled ``approximate`` in the summary for that
reason, and reconciling them against a provider's own metering is a separate
measurement.

A probe is billed differently from how it is measured, and the two must not be
conflated. Teacher forcing decodes nothing, so a probe has no output tokens at all,
but it is billed for the whole prompt plus the continuation, once per
candidate-reference pair. :attr:`CallRecord.input_tokens` therefore carries what the
probe was billed and :attr:`CallRecord.scored_tokens` carries what was actually
measured, which is the continuation alone and is exact because the server returns the
tokens it scored. A cost column built from the first is an accounting fact about the
echo protocol; it is not a statement about the method, and the summary says so in as
many words.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENERATE = "generate"
"""Operation tag for a decoding call."""

SCORE = "score"
"""Operation tag for a teacher-forced probe."""


@dataclass(frozen=True)
class CallRecord:
    """One call to a model.

    Attributes:
        operation: :data:`GENERATE` or :data:`SCORE`.
        model: Model identifier the call named.
        backend: Tag for the serving path, so calls that must never be pooled in one
            statistic can be told apart after the fact.
        latency_s: Wall time of the call, including transport and any waiting.
        input_tokens: Tokens the call is billed for on the way in. For a probe that
            is the prompt *and* the continuation, since the echo protocol sends both.
            Approximate for generation; see the module docstring.
        output_tokens: Decoded tokens. Always zero for a probe, which decodes nothing.
        scored_tokens: Tokens actually teacher-forced. Zero for generation, exact for
            a probe, and the only token count on a probe that says anything about the
            method rather than about the billing.
        n_prompts: Prompts carried by this call. A batched generation is one call and
            several prompts, and a rate is only interpretable if the two are separable.
        ok: Whether the call returned a usable answer.
        error: Short description when it did not.
    """

    operation: str
    model: str
    backend: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    scored_tokens: int = 0
    n_prompts: int = 1
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, one JSONL line's worth."""
        return {
            "operation": self.operation,
            "model": self.model,
            "backend": self.backend,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "scored_tokens": self.scored_tokens,
            "n_prompts": self.n_prompts,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class Price:
    """What a thousand tokens cost, for a spend estimate that is clearly an estimate.

    Attributes:
        input_per_1k: Currency units per thousand prompt tokens.
        output_per_1k: Currency units per thousand completion tokens.
    """

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0

    def of(self, input_tokens: int, output_tokens: int) -> float:
        """Return the estimated cost of a token split."""
        return input_tokens / 1000 * self.input_per_1k + output_tokens / 1000 * self.output_per_1k


class CallLog:
    """The run-level collector: every call, and the aggregates over them.

    Appends are ordinary list appends and the aggregation is computed when asked, so
    recording a call costs nothing that could show up in the latency it is recording.

    Attributes:
        records: Every call, in the order it completed.
        price: Token prices used for the spend estimate.
    """

    records: list[CallRecord]

    def __init__(self, price: Price | None = None) -> None:
        """Initialize an empty log.

        Args:
            price: Token prices. Defaults to zero, which reports a spend of zero
                rather than inventing a number.
        """
        self.records = []
        self.price = price if price is not None else Price()

    def record(self, record: CallRecord) -> None:
        """Append one call."""
        self.records.append(record)

    def __len__(self) -> int:
        """Number of calls recorded."""
        return len(self.records)

    def by_operation(self, operation: str) -> list[CallRecord]:
        """Return every record for one operation.

        Args:
            operation: :data:`GENERATE` or :data:`SCORE`.

        Returns:
            The matching records, in order.
        """
        return [record for record in self.records if record.operation == operation]

    def summary(self) -> dict[str, Any]:
        """Return counts, latency distributions, tokens and an estimated spend.

        Returns:
            A nested mapping: one block per operation plus a total. Latencies are
            reported as a distribution rather than a mean, because a probe path's
            tail is what a batch waits on.
        """
        blocks: dict[str, Any] = {op: _block(self.by_operation(op), self.price) for op in (GENERATE, SCORE)}
        blocks["total"] = _block(self.records, self.price)
        blocks["caveats"] = {
            "generate_tokens": "approximate: promptolution counts whitespace words, not subwords",
            "score_tokens": (
                "input_tokens is what the echo protocol bills, prompt plus continuation, once per "
                "candidate-reference pair. output_tokens is zero because a probe decodes nothing. "
                "scored_tokens is the measurement."
            ),
            "estimated_cost": "a budget rail, not a result: it prices the billing artifact above",
        }
        return blocks

    def to_jsonl(self, path: str | Path) -> Path:
        """Write one JSON object per call.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict()) + "\n")
        return destination


def _block(records: Sequence[CallRecord], price: Price) -> dict[str, Any]:
    """Aggregate one slice of the log.

    Args:
        records: The records to aggregate.
        price: Token prices for the spend estimate.

    Returns:
        Counts, latency percentiles, token totals and estimated cost. Empty-safe:
        an operation that never ran reports zeros rather than absent keys, so a
        summary has the same shape whatever the run did.
    """
    latencies = sorted(record.latency_s for record in records)
    input_tokens = sum(record.input_tokens for record in records)
    output_tokens = sum(record.output_tokens for record in records)
    failures = [record for record in records if not record.ok]
    return {
        "calls": len(records),
        "prompts": sum(record.n_prompts for record in records),
        "failed": len(failures),
        "wall_s": sum(latencies),
        "latency_s": _percentiles(latencies),
        "billed_input_tokens": input_tokens,
        "decoded_output_tokens": output_tokens,
        "scored_tokens": sum(record.scored_tokens for record in records),
        "estimated_cost": price.of(input_tokens, output_tokens),
    }


def _percentiles(latencies: list[float]) -> dict[str, float]:
    """Return the distribution of a sorted latency list.

    Args:
        latencies: Latencies in seconds, ascending.

    Returns:
        Mean, median, p95 and max. All zero for an empty list.
    """
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    index = min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))
    return {
        "mean": statistics.fmean(latencies),
        "p50": statistics.median(latencies),
        "p95": latencies[index],
        "max": latencies[-1],
    }


def approximate_tokens(text: str) -> int:
    """Count tokens the way promptolution does, so one approximation is used throughout.

    Whitespace words, not subwords. Wrong by roughly a third on English, and wrong the
    same way everywhere, which is what makes a client-side total comparable across
    calls even though it is not comparable to a provider's meter.

    Args:
        text: The text to count.

    Returns:
        The word count.
    """
    return len(text.split())
