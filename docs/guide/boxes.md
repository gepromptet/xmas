# A system inside a system

A box runs a whole system as one node of another. Its `reads` map the inner inputs to
outer values, and its `writes` map inner values back out. The inner system names its one
input `page` with `inputs=`; without it, a system's input is `query`:

```python
from xmas import Box

cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
shorten = Box(
    "shorten",
    MAS([cutter], inputs={"page": "text"}, llm=llm),
    reads={"page": "draft"},
    writes={"short": "summary"},
)
first_draft = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
outer = MAS([first_draft], boxes=[shorten], edges=[("writer", "shorten")], llm=llm)

trace = outer.run("Why is the sky blue?")
print(trace.output)  # the cutter's shorter draft
print([(path, record.agent) for path, record in trace.walk()])
# [((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]
```

The box spends from the outer run's budget. Its record has kind `box` and holds the
inner trace, and `trace.walk()` yields every record with the path of boxes around it.
The box's cost and tokens are its inner records' sums, unknown when any of them is, and
a failed box keeps the inner records that finished.
