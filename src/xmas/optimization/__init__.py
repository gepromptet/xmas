"""Optimization: attributors, updaters, and the loops that orchestrate them."""

from xmas.optimization.attribution import UNKNOWN, Attributor, ContrastiveAttributor, UniformAttributor
from xmas.optimization.base import MASOptimizer
from xmas.optimization.cantante import CantanteOptimizer
from xmas.optimization.reference import PrefixSearchOptimizer
from xmas.optimization.updaters import AgentPromptView, PromptUpdater, TopologyUpdater, Updater

__all__ = [
    "UNKNOWN",
    "AgentPromptView",
    "Attributor",
    "CantanteOptimizer",
    "ContrastiveAttributor",
    "MASOptimizer",
    "PrefixSearchOptimizer",
    "PromptUpdater",
    "TopologyUpdater",
    "UniformAttributor",
    "Updater",
]
