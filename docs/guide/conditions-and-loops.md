# Conditions and loops

An edge can carry a condition. Here the JSON judge of
[Answers with a shape](answer-forms.md) sends the draft back while it does not fit, then
publishes it or says sorry. Two more code nodes do the publishing and the apology:

```python
from xmas import Edge, EdgeKind


@tool
def publish(draft: str) -> str:
    """Publish a draft."""
    return f"Published: {draft}"


@tool
def apologize(query: str) -> str:
    """Say sorry when no draft fits."""
    return f"Sorry, no short answer to: {query}"
```

The writer now reads the judge's last verdict, which is empty on its first visit. The
edges say where the run goes after each node:

```python
writer = Agent(
    "writer",
    prompt="Write a short answer to: {query}\nThe judge's last verdict, empty at first: {verdict}",
    inputs=["query", "verdict"],
    output="draft",
)
mas = MAS(
    [writer, judge],
    code_nodes=[Code("count", uses="count_words"), Code("publish", uses="publish"), Code("sorry", uses="apologize")],
    edges=[
        ("writer", "count"),
        ("count", "judge"),
        Edge("judge", "writer", kind=EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
        Edge("judge", "publish", kind=EdgeKind.SWITCH, switch="verdict.action"),
        Edge("judge", "sorry", kind=EdgeKind.ELSE),
        Edge("count", "sorry", kind=EdgeKind.ERROR, errors=("timeout", "outage")),
    ],
    tools=[count_words, publish, apologize],
    llm=llm,
)

print(mas("Why is the sky blue?"))  # "Published: ..." or "Sorry, no short answer to: ..."
```

xmas decides on the judge's fresh verdict. While `fits` is false, the run goes back to
the writer. After three writer visits the loop stops, the record says `capped`, and the
`else` edge runs. When the draft fits, the `switch` edge whose name equals
`verdict.action` is taken. Any other action falls through to `else`. If counting fails
with a timeout or an outage, the run goes on at `sorry` and keeps the failed record.

These are the operators. A spec writes each edge as `a -> b`, with its operator after a
colon:

| Operator | Python | In a spec | Today |
|---|---|---|---|
| `when` | `Edge("a", "b", kind=EdgeKind.WHEN, when="...")` | `a -> b: {when: ...}` | built |
| `else` | `Edge("a", "b", kind=EdgeKind.ELSE)` | `a -> b: else` | built |
| `while` | `Edge("a", "b", kind=EdgeKind.WHILE, when="...", max_visits=3)` | `a -> b: {when: ..., max_visits: 3}` | built |
| `switch` | `Edge("a", "b", kind=EdgeKind.SWITCH, switch="verdict.action")` | `a -> b: {switch: verdict.action}` | built |
| `on error` | `Edge("a", "b", kind=EdgeKind.ERROR, errors=("timeout",))` | `a -> b: {on error: [timeout]}` | built |
| module | `MASModule(name, system, reads=..., writes=...)` | `modules:` | built |
| `for each` | `MAS(..., each={"b": "page in plan.pages"})` | `for each: page in plan.pages` on each node | built |
| `first of` | `Edge("a", "b", kind=EdgeKind.FIRST)` | `a -> b: first of` | built |

A `switch` edge's case is its target's name unless `case="..."` says otherwise. An
`on error` edge takes kinds such as `timeout`, `outage`, `rate_limited`, `money`,
`length_cut`, `bad_format` and `refusal`, or `("any",)`. A budget stop is never caught.

A condition is a small language, parsed when the system is built and never run as code.
It knows paths such as `verdict.fits`, numbers, quoted strings, `true`, `false` and
`null`. It compares with `== != < <= > >= in` and combines with `and`, `or` and `not`.
Its built-ins are `exists(path)`, `len(x)`, `visits`, `visits(node)`, `cost` and
`elapsed`. `verdict.fits = true` is refused with the position of the stray `=`.

Some things are deliberately not operators. A reduce is any node that reads a list. A
filter is `for each` plus `when`. A sample is a code node. Parallel is the default wherever
nothing depends on anything else. A goto is what an edge already is.

A node fed by several edges runs once, after all of them have arrived: a join. When its
edges can never arrive in the same pass, as with `sorry` above, it runs on whichever
arrives: a merge. The check refuses any other mix and names the edges that clash.
