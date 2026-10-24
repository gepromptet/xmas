# Reading a run

The examples read runs of the system of [Conditions and loops](conditions-and-loops.md).
`mas.run(query)` returns the trace; `mas(query)` returns the answer and keeps the trace in
`mas.last_trace`. Suppose the judge never accepts a draft. The trace then shows the loop
and where it stopped:

```python
trace = mas.run("Why is the sky blue?")
print(trace.status)  # ok
print([record.agent for record in trace.records])
# ['writer', 'count', 'judge', 'writer', 'count', 'judge', 'writer', 'count', 'judge', 'sorry']
print(trace["judge"][-1].capped)  # True
print(trace.health.loops)  # {'judge -> writer': {'taken': 2, 'cap_hit': True}}
```

`trace["judge"]` is one record per visit. Each record holds the `inputs` it read, the
`outputs` it wrote and the `rendered_prompt` it sent. It also holds `latency_s`,
`cost_usd`, `tokens`, `finish_reason`, `error_kind` and `served_model`, the model that
actually answered. For a node with conditional edges, `route` names where the visit sent
the run. A step's record has kind `step`.

A run ends with a status: `ok`, `failed` or `over_budget`. A failed run keeps
every record that finished before the failure, and `trace.error_kind` says what kind it
was. Cost is never guessed as zero. It comes from the endpoint's response, or from a
price you give the model (`APILLM(price=TokenPrice(...))`, from `xmas.llms.meta`), or
it stays `None`. A sum with one unknown part is unknown too, so `trace.cost_usd` is
`None` when any record's cost is. A step costs `0.0`, since it makes no model call.

## Health

Every trace carries health signals, read from the records with no model call. They are
information, never rules. Suppose the writer also sent the same draft each time:

```python
print(sorted({(flag.signal, flag.node) for flag in trace.health.flags}))
# [('stuck', 'count'), ('stuck', 'judge'), ('stuck', 'writer')]
```

The signals are `retried`, `bad_format`, `error_routed`, `empty`, `stuck` (the same
output as the node's last visit), `unfilled` (a prompt placeholder with no value), `leak`
(chat-template tokens or reasoning tags in an output), and `long` and `short` (over
three times, or under a third of, the node's median length). A `for each` chain adds
`item_failed` and `item_over_budget` (see [Once per item](chains.md)).
`trace.health.loops` counts each `while` edge, and `trace.health.budget` shows each limit
next to what the run spent. `MAS(..., health=HealthSettings(length_factor=4))` tunes them.
A healthy run's `trace.to_dict()` has no `health` key.

## Budgets

A budget limits every run, and every item of a batch, on its own. A call starts only
while the run is below every limit. This copy of the [quickstart](https://github.com/gepromptet/xmas#readme) may make one call:

```python
from xmas import Budget

thrifty = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_calls=1))
spent = thrifty.run("Why is the sky blue?")
print(spent.status)  # over_budget
print(spent.error)  # BudgetExceeded: calls 1 >= max 1 before call 2
print(spent["intern"][0].outputs)  # the intern's draft is kept
```

`Budget` also takes `max_cost_usd` and `max_wall_s`. A limit of 0 spends nothing. Calls
already in flight may overshoot by one call per branch. `max_parallel_calls` caps how
many model calls are in flight at once, across branches, boxes and chain items. A cost
limit refuses the next call when the last one reported no cost and its model has no
price, because an unknown cost cannot be checked.

## The run log

Every run is logged. The run store is one SQLite file, by default
`~/.local/share/xmas/runs.sqlite`. `XMAS_RUNS` names another file, and
`XMAS_RUNS=:memory:` keeps runs in memory only, so nothing is written to disk. The same
file keeps paused runs, so they resume after a restart (see
[People in the graph](people.md#people-in-the-graph)).

The `xmas` command reads the log from the shell:

```bash
xmas runs              # the logged runs
xmas trace sky-1       # the records of one run, by its run id
```

In Python, `RunStore.trace(run_id)` returns the trace of a logged run.
