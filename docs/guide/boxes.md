# A system inside a system

A module runs a whole system as one node of another. Its `reads` map the inner inputs to
outer values, and its `writes` map inner values back out. The inner system names its one
input `page` with `inputs=`; without it, a system's input is `query`:

```python
from xmas import MASModule

cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
shorten = MASModule(
    "shorten",
    MAS([cutter], inputs={"page": "text"}, llm=llm),
    reads={"page": "draft"},
    writes={"short": "summary"},
)
first_draft = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
outer = MAS([first_draft], modules=[shorten], edges=[("writer", "shorten")], llm=llm)

trace = outer.run("Why is the sky blue?")
print(trace.output)  # the cutter's shorter draft
print([(path, record.agent) for path, record in trace.walk()])
# [((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]
```

The module spends from the outer run's budget. Its record has kind `module` and holds the
inner trace, and `trace.walk()` yields every record with the path of modules around it.
The module's cost and tokens are its inner records' sums, unknown when any of them is, and
a failed module keeps the inner records that finished.
