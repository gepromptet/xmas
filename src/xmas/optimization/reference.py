"""A deliberately trivial optimizer, kept because it is trivial.

It exists to exercise the whole loop end to end, propose through evaluate through
select, with no algorithm worth arguing about in the way. When a real method
misbehaves, this is the control: if the reference optimizer improves a score and
yours does not, the loop is fine and the method is not.
"""

from __future__ import annotations

from xmas.mas import MAS
from xmas.optimization.attribution import Attributor
from xmas.optimization.base import MASOptimizer
from xmas.tasks.base import MASTask

PREFIXES: tuple[str, ...] = (
    "Think step by step.",
    "Be concise and exact.",
    "Show your reasoning, then the answer.",
)
"""Fixed phrases tried against an agent's prompt. Not a method, a fixture."""


class PrefixSearchOptimizer(MASOptimizer):
    """Prepends one of a fixed set of phrases to one agent's prompt, keeps the best.

    Every choice here is intentionally dumb: the candidate set is a constant, the
    agent is picked by round-robin unless an attributor says otherwise, and
    selection is argmax. What it is not dumb about is the plumbing, which is the
    machinery every later optimizer reuses.

    Attributes:
        prefixes: The phrases to try.
    """

    def __init__(
        self,
        mas: MAS,
        task: MASTask,
        prefixes: tuple[str, ...] = PREFIXES,
        attributor: Attributor | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
            prefixes: Phrases to try.
            attributor: Optional credit assignment; when given, the agent with
                the least credit is the one varied.
        """
        super().__init__(mas, task, attributor=attributor)
        self.prefixes = prefixes
        self._round = 0

    def target_agent(self) -> str:
        """Return the agent to vary this step.

        Returns:
            The least-credited agent when credit is available, otherwise the next
            agent in round-robin order, which keeps the sweep deterministic.
        """
        names = list(self.mas.agents)
        scored = {name: values for name, values in self.credit.items() if name in self.mas.agents and values}
        if scored:
            return min(scored, key=lambda name: sum(scored[name]) / len(scored[name]))
        return names[self._round % len(names)]

    def propose(self) -> list[MAS]:
        """Return one candidate per prefix, plus the incumbent.

        The incumbent is included so that a step which finds nothing better keeps
        it rather than being forced to accept a worse candidate.

        Returns:
            The candidates.
        """
        agent = self.target_agent()
        self._round += 1

        base = self.mas.agents[agent].prompt
        candidates = [self.mas]
        seen = {base}
        for prefix in self.prefixes:
            prompt = f"{prefix} {base}".strip()
            if prompt in seen:
                continue
            seen.add(prompt)
            candidates.append(self.mas.copy(prompts={agent: prompt}))
        return candidates

    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Return the highest-scoring candidate.

        Args:
            candidates: The evaluated systems.
            scores: Their scores.

        Returns:
            The best one. Ties go to the earliest, and the incumbent is first, so
            a tie keeps the current system rather than drifting for no gain.
        """
        best = max(range(len(candidates)), key=lambda index: scores[index])
        return candidates[best]
