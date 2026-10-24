# Once per item: `for each`

A `for each` chain runs some nodes once per item of a list. Here a planner lists the
pages of an answer in JSON. Each page is written and counted on its own, and a step
outside the chain adds up the counts. `count_words` is the step of
[Steps and values](steps-and-values.md):

```python
from xmas import ForEach


@tool
def total_words(count_words: list[int | None]) -> int:
    """Add up the word counts of the pages that were written."""
    return sum(count for count in count_words if count is not None)


planner = Agent(
    "planner",
    prompt="Plan a short answer in three pages: {query}",
    output="plan",
    output_type="json",
    schema={"pages": {"type": "list of object", "fields": {"id": {"type": "text"}, "topic": {"type": "text"}}}},
)
page_writer = Agent("page_writer", prompt="Write one page on: {page}", inputs=["page"], output="draft")
deck = MAS(
    [planner, page_writer],
    steps=[Step("count", uses="count_words"), Step("total", uses="total_words")],
    edges=[("planner", "page_writer"), ("page_writer", "count"), ("count", "total")],
    each={"page_writer": ForEach("page", "plan.pages", at_once=2, key="id"), "count": "page in plan.pages"},
    tools=[count_words, total_words],
    llm=llm,
)

trace = deck.run("Why is the sky blue?")
print([(record.agent, record.item_key) for record in trace.records])
# [('planner', None), ('page_writer', 'light'), ('count', 'light'), ('page_writer', 'air'), ...]
print(trace["total"][0].inputs)  # {'count_words': '[4, null, 4]'}: the page 'air' failed
```

Every node of the chain names the same item and list. The first one may add options:
`at_once` caps how many items run at the same time, and `key` names the field that
tells the items apart in the records.

Inside the chain, `page` is one item and `count_words` one number. Outside, each value
the chain writes is a list in item order, however the items finish. When one item
fails, say its model call times out, its place in each list is `None` and the other
items go on. The run still ends `ok`, and health counts the failed item:

```python
print(trace.health.items["page_writer"]["failed"])  # 1, of 3 items
print([flag.signal for flag in trace.health.flags])  # ['item_failed']
```

A budget stop or an out-of-money error still ends the whole run. `item_budget={"cost_usd":
0.05}` on the `ForEach` flags an item that costs more as `item_over_budget`, and never
stops it. A `while` edge inside a chain repairs one item, and its `max_visits` counts per
item. In a spec, each node of the chain carries `for each: page in plan.pages`; the
first may write `{item, in, at once, key, item budget}` instead. Chains do not nest.
