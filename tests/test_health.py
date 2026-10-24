"""Health signals on hand-built records, and the value checks they lean on."""

from __future__ import annotations

import json
from typing import Any

import pytest

from xmas.budget import Budget, RunMeter
from xmas.errors import ErrorKind
from xmas.health import Flag, Health, HealthSettings, summarize
from xmas.trace import AgentRecord, Rejected
from xmas.values import JSON, TEXT, check_value, fits, parse_type, render_value


def rec(
    agent: str,
    step: int,
    output: str | None,
    *,
    prompt: str = "",
    inputs: dict[str, str] | None = None,
    **later: Any,
) -> AgentRecord:
    """Build a record; ``later`` sets the other fields (error_kind, rejected, error, route, capped, item, ...)."""
    outputs = {} if output is None else {"out": output}
    return AgentRecord(agent, step, prompt, prompt, inputs or {}, outputs, **later)


RECORDS = (
    rec("w", 0, "a" * 9, prompt="Write {query}", inputs={"query": "cats"}),
    rec("judge", 1, '{"fits": false}', rejected=(Rejected("Sure.", "<root>: Invalid JSON"),), route=("w",)),
    rec("w", 2, "b" * 9),
    rec("judge", 3, '{"fits": false}', route=("w",)),
    rec("w", 4, "c" * 9),
    rec("judge", 5, '{"fits": true}', route=("pub",), capped=True),
    rec("w", 6, "d" * 27),  # 3 times the median of 9: not long
    rec("w", 7, "e" * 28),  # just beyond: long
    rec("w", 8, "f" * 3),  # the median divided by 3: not short
    rec("w", 9, "g" * 2),  # just under: short
    rec("pub", 10, " \n", prompt="Publish {draft} for {reader}", inputs={"draft": ""}),
    rec("count", 11, None, error_kind=ErrorKind.BAD_FORMAT, error="count: the answer was not JSON", route=("sorry",)),
    rec("few", 12, "x"),  # two samples: under min_samples, no median
    rec("few", 13, "y" * 100),
    rec("leaky", 14, "Done <|im_end|>"),
)


def spent_meter() -> RunMeter:
    """A meter that counted 4 calls and 0.5 USD against limits of 10 and 1.0."""
    meter = RunMeter(Budget(max_calls=10, max_cost_usd=1.0))
    meter.calls, meter.cost_usd = 4, 0.5
    return meter


def test_every_signal_flags_its_record_with_a_detail() -> None:
    health = summarize(RECORDS, meter=spent_meter(), settings=HealthSettings(), while_edges=[("judge", "w")])
    assert [(flag.signal, flag.node, flag.step) for flag in health.flags] == [
        ("retried", "judge", 1),
        ("stuck", "judge", 3),
        ("long", "w", 7),
        ("short", "w", 9),
        ("empty", "pub", 10),
        ("unfilled", "pub", 10),
        ("bad_format", "count", 11),
        ("error_routed", "count", 11),
        ("leak", "leaky", 14),
    ]
    by_signal = {flag.signal: flag for flag in health.flags}
    assert by_signal["retried"] == Flag("retried", "judge", 1, "1 answer(s) rejected: <root>: Invalid JSON")
    assert by_signal["stuck"].detail == "the same output as step 1"
    assert by_signal["long"].detail == "28 characters, the node's median is 9"
    assert by_signal["unfilled"].detail == "placeholders without a value: reader"
    assert by_signal["error_routed"].detail == "bad_format: count: the answer was not JSON"
    assert by_signal["leak"].detail == "the output holds '<|im_end|>'"

    data = health.to_dict()
    assert list(data) == ["flags", "loops", "budget"]
    assert data["flags"][0] == {"signal": "retried", "node": "judge", "step": 1, "detail": by_signal["retried"].detail}
    assert json.loads(json.dumps(data)) == data


def test_loops_and_budget_come_from_the_routes_and_the_meter() -> None:
    meter = spent_meter()
    health = summarize(RECORDS, meter=meter, while_edges=[("judge", "w"), ("w", "judge")])
    assert health.loops == {"judge -> w": {"taken": 2, "cap_hit": True}}  # w -> judge was never taken
    assert summarize(RECORDS).loops == {}
    assert health.budget == {"calls": {"spent": 4, "limit": 10}, "cost_usd": {"spent": 0.5, "limit": 1.0}}
    assert summarize(RECORDS, meter=RunMeter(Budget(max_wall_s=60))).budget["wall_s"]["limit"] == 60
    assert summarize(RECORDS, meter=RunMeter()).budget == {}  # no budget, no limits to report
    assert summarize(RECORDS).budget == {}  # no meter, nothing spent to report


def test_settings_tune_min_samples_length_and_leaks() -> None:
    two = summarize(RECORDS, settings=HealthSettings(min_samples=2))
    assert ("short", "few", 12) in [(flag.signal, flag.node, flag.step) for flag in two.flags]
    wide = summarize(RECORDS, settings=HealthSettings(length_factor=4.5))
    assert not {flag.signal for flag in wide.flags} & {"long", "short"}
    assert [flag.detail for flag in summarize([rec("a", 0, "<think>hm</think> yes")]).flags] == [
        "the output holds '<think>'"
    ]
    secret = [rec("a", 0, "the SECRET is out")]
    assert summarize(secret).flags == ()
    assert [flag.detail for flag in summarize(secret, settings=HealthSettings(leak=("SECRET",))).flags] == [
        "the output holds 'SECRET'"
    ]


def test_an_unrouted_failure_is_not_flagged_as_routed() -> None:
    """It ended the run, and the trace's error says so."""
    unrouted = rec("fetch", 0, None, error_kind=ErrorKind.OUTAGE, error="ConnectionError: the server is down")
    assert summarize([unrouted]).flags == ()


def test_unfilled_counts_only_placeholders_that_name_a_value() -> None:
    """A literal ``{answer}`` in a prompt is prompt text unless the system has a value of that name."""
    prompt = 'Reply as {"answer": ...} with {answer} in {query}'
    literal = [rec("solver", 0, "42", prompt=prompt, inputs={"query": "q"})]
    assert summarize(literal, values={"query", "out"}).flags == ()
    assert [(flag.signal, flag.detail) for flag in summarize(literal, values={"query", "answer"}).flags] == [
        ("unfilled", "placeholders without a value: answer")
    ]


def test_a_healthy_run_reports_nothing() -> None:
    healthy = summarize([rec("w", 0, "a draft", prompt="Write {query}", inputs={"query": "cats"}), rec("j", 1, "ok")])
    assert healthy == Health()
    assert healthy.to_dict() == {}


def test_items_and_item_budgets_are_flagged_per_chain() -> None:
    def page(
        agent: str, step: int, output: str | None, item: int, cost: float | None = 0.02, **later: Any
    ) -> AgentRecord:
        return rec(agent, step, output, item=item, item_key=f"page-{item}", latency_s=50.0, cost_usd=cost, **later)

    records = [
        page("drafter", 0, "draft A", 0),
        page("checker", 1, "fits", 0),
        page("drafter", 2, "draft A", 1),  # the same text as item 0's draft: another item, not stuck
        page("checker", 3, None, 1, error_kind=ErrorKind.OUTAGE, error="ConnectionError: down", cost=None),
        page("drafter", 4, "draft B", 2),
        page("checker", 5, "refit", 2, route=("drafter",)),
        page("drafter", 6, "draft B", 2),  # the same text within item 2: stuck
        page("checker", 7, "fits", 2),
        rec("assemble", 8, "deck"),
    ]
    budget = {"cost_usd": 0.05, "wall_s": 120.0}
    chains = {"drafter": (["drafter", "checker"], budget), "captioner": (["captioner"], {})}
    health = summarize(records, chains=chains)
    assert [(flag.signal, flag.node, flag.step, flag.detail) for flag in health.flags] == [
        ("item_failed", "checker", 3, "item 1 (page-1) failed: outage: ConnectionError: down"),
        ("stuck", "drafter", 6, "the same output as step 4"),
        ("item_over_budget", "checker", 7, "item 2 (page-2) cost 0.08 USD, over its budget of 0.05"),
        ("item_over_budget", "checker", 7, "item 2 (page-2) took 200 s, over its budget of 120"),
    ]
    assert health.items == {
        "drafter": {"items": 3, "failed": 1, "cost_usd": [0.04, None, pytest.approx(0.08)]},
        "captioner": {"items": 0, "failed": 0, "cost_usd": []},
    }
    data = summarize(records[:2], chains={"drafter": chains["drafter"]}).to_dict()
    assert data == {"items": {"drafter": {"items": 1, "failed": 0, "cost_usd": [0.04]}}}
    assert json.loads(json.dumps(data)) == data

    alone = summarize(records)  # no chain given: item failures still flag, no item report
    assert [flag.signal for flag in alone.flags] == ["item_failed", "stuck"]
    assert alone.items == {}
    assert "items" not in alone.to_dict()
    assert summarize(records[:2]).to_dict() == {}


def test_list_values_check_and_fit() -> None:
    texts, nested = parse_type("list of text"), parse_type("list of list of json")
    assert check_value(["a", None, "c"], texts) is None  # a gathered list holds null where an item failed
    assert check_value(["a", "b", 3], texts) == "expected text at 2, got int"
    assert check_value("a", texts) == "expected list of text, got str"
    assert check_value([None, [None, {"n": 1}]], nested) is None
    assert check_value([[1], "x"], nested) == "expected list of json at 1, got str"
    assert check_value([[1], [{"n": {2}}]], nested) == "not JSON at 1.0.n: set"
    assert check_value(3, TEXT) == "expected text, got int"

    assert fits(texts, JSON)
    assert fits(nested, JSON)
    assert fits(texts, texts)
    assert not fits(JSON, texts)
    assert not fits(TEXT, JSON)
    assert not fits(texts, nested)

    assert render_value(["a", None], texts) == '["a", null]'
    assert render_value([["é"]], nested) == '[["é"]]'
    assert render_value(["a"], texts, present=False) == ""
