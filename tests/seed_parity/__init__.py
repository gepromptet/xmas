"""Vendored seed implementation, for parity checking only. See README.md.

The vendored module is untyped, as the seed repositories are. This package gives
it one annotated entry point so that strict type checking stops at the boundary
instead of every parity test having to silence it.
"""
# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Mapping

from tests.seed_parity.seed_attributer import AbsoluteAttributer, AgentPromptBatch

__all__ = ["seed_attribute"]


def seed_attribute(
    llm: object,
    agent_names: list[str],
    agent_var_mapping: Mapping[str, Mapping[str, object]],
    system_scores: list[list[float]],
    intermediate_results: list[list[dict[str, str]]],
    *,
    group_size: int,
    seed: int = 42,
) -> dict[str, list[list[float]]]:
    """Run the vendored seed attributer and return its credit, unchanged.

    Args:
        llm: The judging model.
        agent_names: Agents the seed should attribute over.
        agent_var_mapping: The seed's per-agent input/output variable map.
        system_scores: ``[run][query]`` system scores.
        intermediate_results: ``[run][query]`` task state dicts.
        group_size: Executions compared per judging call.
        seed: Seed for the shuffle.

    Returns:
        ``[run][query]`` credit per agent, exactly as the seed computed it.
    """
    attributer = AbsoluteAttributer(
        llm=llm,
        agent_var_mapping=agent_var_mapping,
        group_size=group_size,
        seed=seed,
    )
    return dict(attributer.attribute(AgentPromptBatch(agent_names), system_scores, intermediate_results))
