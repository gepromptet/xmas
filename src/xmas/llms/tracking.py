"""Per-step record for the anytime curves.

The run's headline is not a final score, it is a curve: best dev score against
cumulative wall-clock, and against cumulative billed tokens, at equal budget per
arm. That only works if a snapshot is taken at every evaluation rather than
reconstructed at the end, because the thing being compared is the shape of the
climb and not its endpoint.

Generation and probe counts are kept apart throughout. They are different
primitives billed in different ways -- a probe decodes nothing but is billed for
prompt plus continuation -- so pooling them into one token column would make the
one number the paper cares about meaningless.

Client-side token counts are approximate: promptolution counts by splitting on
whitespace. The serving side's metering is authoritative; pass the measured
client-to-billed factor as `gateway_factor` and these stay labelled estimates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xmas.llms.call_log import CallLog


class AnytimeTracker:
    """One row per evaluation, carrying everything the curves need.

    Attributes:
        arm: Which arm this is, so two files can be plotted against each other.
        rows: The snapshots, in order.
    """

    def __init__(self, telemetry: CallLog, arm: str, gateway_factor: float = 1.0) -> None:
        """Initialize against the CallLog the run's models record into."""
        self.telemetry = telemetry
        self.arm = arm
        self.gateway_factor = gateway_factor
        """Client-to-billed token factor, measured against the serving side's own
        metering. 1.0 means the client counts are reported as they are."""
        self.rows: list[dict[str, Any]] = []
        self._started = time.perf_counter()
        self._best = float("-inf")

    def record(
        self,
        step: int,
        scores: Sequence[float],
        ess: Sequence[float] | None = None,
        refreshed: bool = False,
    ) -> dict[str, Any]:
        """Snapshot the run after one evaluation.

        Args:
            step: Evaluation index, zero-based.
            scores: This evaluation's candidate scores.
            ess: Per-candidate effective sample sizes, when the arm has them.
            refreshed: Whether the reference pool was regenerated at this step.

        Returns:
            The row just appended.
        """
        finite = [float(v) for v in scores if v == v]
        self._best = max([self._best, *finite]) if finite else self._best

        generated = [r for r in self.telemetry.records if r.operation == "generate"]
        probed = [r for r in self.telemetry.records if r.operation == "score"]
        client_in = sum(r.input_tokens for r in generated) + sum(r.input_tokens for r in probed)

        row: dict[str, Any] = {
            "arm": self.arm,
            "step": step,
            "best_score": self._best if self._best > float("-inf") else None,
            "step_best": max(finite) if finite else None,
            "wall_s": round(time.perf_counter() - self._started, 4),
            "gen_calls": len(generated),
            "gen_input_tokens": sum(r.input_tokens for r in generated),
            "gen_output_tokens": sum(r.output_tokens for r in generated),
            "probe_calls": len(probed),
            "probe_billed_input_tokens": sum(r.input_tokens for r in probed),
            "scored_tokens": sum(r.scored_tokens for r in probed),
            "client_total_tokens": client_in + sum(r.output_tokens for r in generated),
            "estimated_gateway_tokens": round(
                (client_in + sum(r.output_tokens for r in generated)) * self.gateway_factor
            ),
            "mean_ess": (sum(ess) / len(ess)) if ess else None,
            "pool_refreshed": refreshed,
        }
        self.rows.append(row)
        return row

    def to_jsonl(self, path: str | Path) -> Path:
        """Write the rows, one JSON object per line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")
        return Path(path)
