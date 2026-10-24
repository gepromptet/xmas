# Steps and values

The guide grows one small system, page by page. Each page uses the names the pages
before it define, and `llm` is the model of the
[quickstart](https://github.com/gepromptet/xmas#readme) in the README.

## Code between agents: steps

A writer drafts, a step counts the words, and a judge reads both. The count is plain Python, so it needs no model call.

A step runs a function from the system's `tools=` list. The function needs a type hint on
every parameter and a docstring.

```python
from xmas import MAS, Agent, Step
from xmas.tools import tool


@tool
def count_words(draft: str) -> int:
    """Count the words of a draft."""
    return len(draft.split())


writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
judge = Agent(
    "judge",
    prompt="Question: {query}\nDraft, {count_words} words: {draft}\nIs it a good answer? Say why in one sentence.",
    inputs=["query", "draft", "count_words"],
    output="review",
)
mas = MAS(
    [writer, judge],
    steps=[Step("count", uses="count_words")],
    edges=[("writer", "count"), ("count", "judge")],
    tools=[count_words],
    llm=llm,
)

print(mas("Why is the sky blue?"))  # the judge's review
print([record.agent for record in mas.last_trace.records])  # ['writer', 'count', 'judge']
```

A step reads its parameters by name, here `draft`. It writes one value named like the
function, here `count_words`. That value is text when the function returns `str`, and
JSON otherwise. `Step(..., reads={"draft": "answer"}, writes={"count_words": "words"})`
renames both sides.

The same function can also be a tool: `Agent(..., tools=["count_words"])` lets the
model call it while it answers. As a step, the graph decides when it runs; as a tool,
the model does. An MCP tool from `MCPTools` works in both places too.

In Python, an agent's reads are `inputs` and its write is `output`. The saved spec (see
[Bundles](bundles.md)) calls them `reads` and `writes`.

## Mistakes found before any call

xmas checks the wiring when you build the system, before any model call. It lists every
problem at once, so one edit fixes them all. This judge has three:

```python
judge_with_typos = Agent(
    "judge",
    prompt="Draft: {draft}\nIt has {count_words} words. The style guide: {style}",
    inputs=["draft", "notes"],
    output="verdict",
)
try:
    MAS(
        [writer, judge_with_typos],
        steps=[Step("count", uses="count_word")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )
except ValueError as problems:
    print(problems)
```

The message names each one:

```text
The system does not check (3 problems):
- agent 'judge': prompt placeholders ['count_words', 'style'] are not among its reads ['draft', 'notes']
- Step 'count' uses unknown callable 'count_word'; available: ['count_words'].
- 'judge' reads 'notes', which is neither an input nor written by any node
```

The check also refuses two writers of one value, a JSON value sent to a function that
wants text, and a condition that does not parse.
