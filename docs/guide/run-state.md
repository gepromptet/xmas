# Several values in, several out: `run_state`

`run(query)` takes one text. `run_state` starts a run from all of a system's inputs by
name, and `outputs=` names the values the trace hands back. Here are the pages of
[Once per item](chains.md) again, started from a plan you already have:

```python
pages = MAS(
    [page_writer],
    code_nodes=[Code("count", uses="count_words"), Code("total", uses="total_words")],
    edges=[("page_writer", "count"), ("count", "total")],
    each={"page_writer": ForEach("page", "plan.pages", key="id"), "count": "page in plan.pages"},
    inputs={"plan": "json"},
    outputs={"total_words": "json", "count_words": "list of json"},
    tools=[count_words, total_words],
    llm=llm,
)

heard = []
trace = pages.run_state({"plan": {"pages": [{"id": "light"}, {"id": "eye"}]}}, on_event=heard.append)
print(trace.outputs)  # {'total_words': 8, 'count_words': [4, 4]}
print(trace.output)  # 8: the first output, which a task scores
print(sorted({event.kind for event in heard}))  # ['end', 'item_end', 'item_start', 'start']
```

A missing, extra or mistyped input raises one `ValueError` that lists them all, before
any call. `on_event` hears each visit's start and end and each item's, with the node,
the item and a status. It runs on the run's threads, so keep it quick.

With `outputs=`, several branches may end the run, and the run ends once all of them
have. Without it, `output` is the last terminal's write, as before. The trace of a
version 2 run carries a `run_id`, and its `graph_id` is the system's `describe()["id"]`.

Three more things a node can do. `MAS(..., retry={"count": {"outage": 1}})` runs a node
again after an outage, a timeout or a rate limit. Its one record counts the `retries`
and the cost of every attempt. A node may read two values that no node writes: `error`,
the `{kind, message, node}` of the failure an `on error` edge routed to it, and `run`,
what the run has spent so far. `Agent(..., inputs={"slide": "page"})` fills the
placeholder `{slide}` from the value `page`.
