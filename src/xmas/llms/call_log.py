"""Call-level statistics for the LLM layer: a frozen record per call and a collector.

Two honesty notes. promptolution counts tokens by splitting on whitespace, so the token
counts of a generation are an approximation (an undercount: about 1.3 subword tokens per
word on English). A probe decodes nothing but is billed for prompt plus continuation, so
``input_tokens`` carries what it was billed and ``scored_tokens`` what it measured.
"""

from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENERATE = "generate"
"""Operation tag for a decoding call."""

SCORE = "score"
"""Operation tag for a teacher-forced probe."""


def timestamp() -> str:
    """Return the current time as ISO 8601 with an explicit UTC offset."""
    return datetime.now(UTC).isoformat()


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
        call_id: Random UUID4 naming this record, so a row can be cited and a
            duplicate (the same log appended twice) can be spotted.
        ts: When the call started, ISO 8601 with an explicit offset.
        prompt: The user prompt sent, for a single-prompt call with text capture on.
        system_prompt: The system prompt sent alongside :attr:`prompt`.
        response: The text returned for :attr:`prompt`. ``None`` on a failed call.
        params: What the call was sent besides the messages: ``max_tokens`` and every
            ``call_kwargs`` entry except ``extra_headers``, which can carry credentials.
            Recorded whether or not text capture is on.
        response_id: The provider's completion id for a single-prompt call, when the
            client exposes one. Recorded whether or not text capture is on.
        prompts: A batched call's prompts, in order, with text capture on. A batch
            stays one record, so exactly one of :attr:`prompt` and :attr:`prompts`
            is set on a captured generation.
        system_prompts: A batched call's system prompts, aligned to :attr:`prompts`.
        responses: A batched call's responses, aligned to :attr:`prompts`.
        response_ids: A batched call's completion ids, aligned to :attr:`prompts`.
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
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=timestamp)
    prompt: str | None = None
    system_prompt: str | None = None
    response: str | None = None
    params: dict[str, Any] | None = None
    response_id: str | None = None
    prompts: list[str] | None = None
    system_prompts: list[str] | None = None
    responses: list[str] | None = None
    response_ids: list[str | None] | None = None

    def exchanges(self) -> list[dict[str, str | None]]:
        """Return the captured text one prompt at a time, whatever the call's shape.

        A single-prompt call and a batch store their text in different fields; this
        is the one place a reader has to know that.

        Returns:
            One mapping per prompt with ``system_prompt``, ``prompt``, ``response`` and
            ``response_id``. Empty when the text was not captured.
        """
        if self.prompts is not None:
            count = len(self.prompts)
            absent: list[str | None] = [None] * count
            system_prompts: list[str | None] = list(self.system_prompts) if self.system_prompts else absent
            responses: list[str | None] = list(self.responses) if self.responses else absent
            response_ids = self.response_ids or absent
            return [
                {"system_prompt": system, "prompt": prompt, "response": response, "response_id": response_id}
                for system, prompt, response, response_id in zip(
                    system_prompts, self.prompts, responses, response_ids, strict=True
                )
            ]
        if self.prompt is not None:
            return [
                {
                    "system_prompt": self.system_prompt,
                    "prompt": self.prompt,
                    "response": self.response,
                    "response_id": self.response_id,
                }
            ]
        return []

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
            "call_id": self.call_id,
            "ts": self.ts,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "response": self.response,
            "params": self.params,
            "response_id": self.response_id,
            "prompts": self.prompts,
            "system_prompts": self.system_prompts,
            "responses": self.responses,
            "response_ids": self.response_ids,
        }


@dataclass
class CallLog:
    """The run-level collector: every call, and the aggregates over them.

    Appends are ordinary list appends and the aggregation is computed when asked, so
    recording a call costs nothing that could show up in the latency it is recording.

    Attributes:
        records: Every call, in the order it completed.
        capture_text: Whether backends put the prompt and response text on each record.
        input_price_per_1k: Currency units per thousand prompt tokens; zero reports zero spend.
        output_price_per_1k: Currency units per thousand completion tokens.
    """

    records: list[CallRecord] = field(default_factory=list)
    capture_text: bool = False
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0

    def record(self, record: CallRecord) -> None:
        """Append one call."""
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        """Return counts, latency distributions, tokens and an estimated spend.

        Returns:
            A nested mapping: one block per operation plus a total. Latencies are
            reported as a distribution rather than a mean, because a probe path's
            tail is what a batch waits on.
        """
        blocks = {op: aggregate(self, [r for r in self.records if r.operation == op]) for op in (GENERATE, SCORE)}
        blocks["total"] = aggregate(self, self.records)
        return blocks

    def to_jsonl(self, path: str | Path, append: bool = False) -> Path:
        """Write one JSON object per call.

        Args:
            path: Destination file.
            append: Add to the end of an existing file instead of replacing it. Each
                record carries its ``call_id``, so appending the same log twice is
                detectable afterwards.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a" if append else "w", encoding="utf-8") as handle:
            for record in self.records:
                # ``default=str`` keeps a non-JSON call kwarg (a response-format
                # class, say) from losing the whole dump.
                handle.write(json.dumps(record.to_dict(), default=str) + "\n")
        return destination


def aggregate(log: CallLog, records: list[CallRecord]) -> dict[str, Any]:
    """Aggregate one slice of the log; an operation that never ran reports zeros, not absent keys."""
    latencies = sorted(record.latency_s for record in records)
    input_tokens = sum(record.input_tokens for record in records)
    output_tokens = sum(record.output_tokens for record in records)
    # Latency as a distribution, not a mean: a probe path's tail is what a batch waits on.
    latency = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if latencies:
        latency = {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "p95": latencies[round(0.95 * (len(latencies) - 1))],
            "max": latencies[-1],
        }
    return {
        "calls": len(records),
        "prompts": sum(record.n_prompts for record in records),
        "failed": sum(not record.ok for record in records),
        "wall_s": sum(latencies),
        "latency_s": latency,
        "billed_input_tokens": input_tokens,
        "decoded_output_tokens": output_tokens,
        "scored_tokens": sum(record.scored_tokens for record in records),
        "estimated_cost": input_tokens / 1000 * log.input_price_per_1k + output_tokens / 1000 * log.output_price_per_1k,
    }


def approximate_tokens(text: str) -> int:
    """Count whitespace words, the same undercount promptolution uses.

    Comparable across calls, not to a provider's meter.
    """
    return len(text.split())
