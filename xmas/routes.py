"""Route analysis: whether a version 2 node with several incoming edges merges or joins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

from xmas.edges import Edge, EdgeKind
from xmas.graph import END, START

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ROUTE_CAP",
    "Join",
    "RouteGraph",
]


ROUTE_CAP = 256
"""Most route combinations kept per node; past it a merge or join cannot be proved."""

Point = str | tuple[str, str, str]
"""A node of the route graph: a node name, or a route ``(node, decision, label)``."""

FAIL, CHOOSE, OK = "fail", "choose", ""

MULTI_INPUT = 2
"""The fewest incoming points that make a node a merge or a join."""


@dataclass(frozen=True)
class Join:
    """How a version 2 node with several incoming edges runs.

    ``groups`` holds the sources of the incoming edges; the edges of one group
    are alternatives. A ``"merge"`` has one group, whose edges never arrive in
    the same pass: it runs once on whichever arrives. A ``"join"`` has several
    groups that each arrive on every pass reaching it: it waits for one edge of
    every group, once per pass (a barrier). A join has one relay per group:
    ``None`` when the group is one plain edge from a node without error edges,
    so the barrier waits for that node, else the no-op node
    (``__in__.<node>.<k>``) the group's edges enter.
    """

    kind: str
    groups: tuple[tuple[str, ...], ...]
    relays: tuple[str | None, ...] = ()


class RouteGraph:
    """One pass of a version 2 topology, every route of every decision a node of its own.

    ``while`` edges are left out (``max_visits`` forces their exit). A node with
    ``when``, ``while``, ``switch`` or ``else`` edges makes one decision whose
    routes are those edges; a node with ``on error`` edges makes another, its
    plain and deciding edges hanging under an ``ok`` route. A node's choice sets
    are the minimal sets of routes under which it runs in one pass; two nodes
    are alternatives when every pair of their sets takes different routes of
    some decision, which is exact because a decision runs at most once a pass.
    The edges other than ``while`` edges must not form a cycle.
    """

    def __init__(self, nodes: Sequence[str], edges: Sequence[Edge]) -> None:
        """Build the route graph and every node's choice sets."""
        self.nodes = list(nodes)
        self.graph: nx.DiGraph[Any] = nx.DiGraph()
        self.graph.add_nodes_from([START, *self.nodes, END])
        failing = {edge.source for edge in edges if edge.kind is EdgeKind.ERROR}
        self.loops = [(edge.source, edge.target) for edge in edges if edge.kind is EdgeKind.WHILE]
        for edge in edges:
            if edge.kind is not EdgeKind.WHILE:
                self.add_routes(edge, failing)
        self.add_ends({edge.source for edge in edges if edge.kind is not EdgeKind.ERROR}, failing)
        self.choices: dict[Point, set[frozenset[Point]] | None] = {START: {frozenset()}}
        for point in nx.topological_sort(self.graph):
            if point != START:
                self.choices[point] = self.choice_sets(point)
        self.reach: dict[tuple[Point, tuple[Point, ...]], bool] = {}

    def add_routes(self, edge: Edge, failing: set[str]) -> None:
        """Add one edge, through a route unless plain, under its source's ``ok`` route when it has error edges."""
        head: Point = edge.source
        if edge.source in failing and edge.kind is not EdgeKind.ERROR:
            head = (edge.source, FAIL, OK)
            self.graph.add_edge(edge.source, head, kind="route")
        if edge.kind is EdgeKind.PLAIN:
            self.graph.add_edge(head, edge.target, kind="plain")
            return
        route = (edge.source, FAIL if edge.kind is EdgeKind.ERROR else CHOOSE, edge.target)
        self.graph.add_edge(head, route, kind="route")
        self.graph.add_edge(route, edge.target, kind="plain")

    def add_ends(self, sources: set[str], failing: set[str]) -> None:
        """Start each node nothing enters from START, and end each node that is no source (``sources``) at END."""
        for name in self.nodes:
            if self.graph.in_degree(name) == 0:
                self.graph.add_edge(START, name, kind="plain")
            if name in sources:
                continue
            if name in failing:  # only error edges: a visit that succeeds ends the run
                ok = (name, FAIL, OK)
                self.graph.add_edge(name, ok, kind="route")
                self.graph.add_edge(ok, END, kind="plain")
            else:
                self.graph.add_edge(name, END, kind="plain")

    def choice_sets(self, point: Point) -> set[frozenset[Point]] | None:
        """Return the minimal route sets under which ``point`` runs, ``None`` past :data:`ROUTE_CAP`."""
        extra = {point} if isinstance(point, tuple) else set()
        sets: set[frozenset[Point]] = set()
        for parent in self.graph.predecessors(point):
            known = self.choices.get(parent)
            if known is None:
                return None
            sets |= {choice | extra for choice in known}
        minimal = {choice for choice in sets if not any(other < choice for other in sets)}
        return minimal if len(minimal) <= ROUTE_CAP else None

    def exclusive(self, first: Point, second: Point) -> bool:
        """Return whether two points never both run in one pass; ``False`` when that cannot be proved."""
        left, right = self.choices.get(first), self.choices.get(second)
        if left is None or right is None:
            return False
        return all(clash(one, other) for one in left for other in right)

    def failed_before(self, name: str) -> frozenset[str]:
        """Return the nodes through whose error routes every pass that runs ``name`` goes."""
        sets = self.choices.get(name)
        if not sets:
            return frozenset()
        failed = [
            {point_node(route) for route in choice if not every_pass(route) and route[1] == FAIL} for choice in sets
        ]
        return frozenset(set.intersection(*failed))

    def reaches(self, point: Point, group: tuple[Point, ...]) -> bool:
        """Return whether every pass that runs ``point`` also runs a member of ``group``."""
        key = (point, group)
        if key not in self.reach:
            plain: list[Point] = []
            routes: list[Point] = []
            for child in self.graph.successors(point):
                (plain if self.graph.edges[point, child]["kind"] == "plain" else routes).append(child)
            self.reach[key] = (
                point in group
                or any(self.reaches(child, group) for child in plain)
                or (bool(routes) and all(self.reaches(child, group) for child in routes))
            )
        return self.reach[key]

    def loop_bodies(self) -> list[tuple[str, str, set[str]]]:
        """Return each loop as ``(source, target, the nodes it runs again)``."""
        bodies = []
        for source, target in self.loops:
            inside = {target, *nx.descendants(self.graph, target)} & {source, *nx.ancestors(self.graph, source)}
            bodies.append((source, target, {point for point in inside if isinstance(point, str)}))
        return bodies

    def classify(self) -> tuple[dict[str, Join], dict[str, str]]:
        """Return each multi-input node's merge or join, and the refusal of each node that is neither."""
        joins: dict[str, Join] = {}
        refusals: dict[str, str] = {}
        idom: dict[Any, Any] | None = None
        for name in [*self.nodes, END]:
            points: list[Point] = list(self.graph.predecessors(name))
            if len(points) < MULTI_INPUT:
                continue
            if any(self.choices.get(point) is None for point in points):
                refusals[name] = (
                    f"node {name!r}: more than {ROUTE_CAP} route combinations reach it, so whether it merges "
                    "or joins cannot be proved; put part of the system in a box"
                )
                continue
            groups: list[list[Point]] = []
            for point in points:
                fits = [group for group in groups if all(self.exclusive(point, other) for other in group)]
                if fits:
                    fits[0].append(point)
                else:
                    groups.append([point])
            named = tuple(tuple(point_node(point) for point in group) for group in groups)
            if len(groups) == 1:
                joins[name] = Join("merge", named)
                continue
            idom = idom if idom is not None else nx.immediate_dominators(self.graph, START)
            problem = self.join_problem(name, groups, idom[name])
            if problem:
                refusals[name] = problem
                continue
            # One plain edge from a node is waited on at the node; a route (a
            # decision's, or the success route of a node with error edges), the
            # start or several alternatives get a relay.
            relays = tuple(
                None if len(group) == 1 and group[0] != START and isinstance(group[0], str) else f"__in__.{name}.{k}"
                for k, group in enumerate(groups)
            )
            joins[name] = Join("join", named, relays)
        return joins, refusals

    def join_problem(self, name: str, groups: list[list[Point]], dominator: Point) -> str | None:
        """Say why ``name`` cannot wait for one edge of every group, or return ``None``."""

        def edges(group: list[Point]) -> str:
            return " or ".join(f"{point_node(point)} -> {name}" for point in group)

        for group in groups:
            if not self.reaches(dominator, tuple(group)):
                other = next(rest for rest in groups if rest is not group)
                return (
                    f"node {name!r} is neither a merge nor a join: {edges(group)} does not run on every pass "
                    f"that runs {edges(other)}; join the alternatives in one node first"
                )
        for source, target, inside in self.loop_bodies():
            for point in (point for group in groups for point in group):
                if name not in inside and every_pass(point) and point_node(point) in inside:
                    return (
                        f"node {name!r} cannot wait for {point_node(point)} -> {name}, which leaves the loop "
                        f"{target} .. {source} on every pass; leave a loop through its else edge"
                    )
        return None


def point_node(point: Point) -> str:
    """Return the node a point of the route graph belongs to."""
    return point if isinstance(point, str) else point[0]


def every_pass(point: Point) -> bool:
    """Return whether a point is a node, or the route its node takes whenever it succeeds."""
    return isinstance(point, str) or point[1:] == (FAIL, OK)


def clash(first: frozenset[Point], second: frozenset[Point]) -> bool:
    """Return whether two route sets take different routes of one decision."""
    return any(
        a[:2] == b[:2] and a[2] != b[2] for a in first for b in second if isinstance(a, tuple) and isinstance(b, tuple)
    )
