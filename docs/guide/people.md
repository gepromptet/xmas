# People, Stop and races

## People in the graph

A person can be a node too. A `Human` asks a question, and the run waits for the answer.
Here an editor approves the draft before it is published, with the draft in the question.
`publish` is the step of [Conditions and loops](conditions-and-loops.md):

```python
from xmas import END, Human

editor = Human(
    "editor",
    "approve",
    reads=["draft"],
    writes="approval",
    question="Publish this answer? {draft}",
    labels={"yes": "Publish", "no": "Hold it"},
)
gated = MAS(
    [Agent("writer", prompt="Write a short answer to: {query}", output="draft")],
    steps=[Step("publish", uses="publish")],
    humans=[editor],
    edges=[
        ("writer", "editor"),
        Edge("editor", "publish", kind=EdgeKind.WHEN, when="approval.approved"),
        Edge("editor", END, kind=EdgeKind.ELSE),
    ],
    tools=[publish],
    llm=llm,
)
```

Run it under a `run_id` of your choice. It returns paused, and the answer resumes it:

```python
trace = gated.run_state({"query": "Why is the sky blue?"}, run_id="sky-1")
print(trace.status, [question.id for question in trace.pause.questions])  # paused ['sky-1/editor/1']
print(trace.pause.questions[0].options)  # ('Publish', 'Hold it')
done = gated.resume("sky-1", {"sky-1/editor/1": {"approved": True, "who": "tom"}})
print(done.status, done.output)  # ok Published: Sunlight scatters off air.
```

The run pauses only when nothing else is left to do. Other branches run on and end
first, and all open questions are asked together in `trace.pause`. `gated.answer(run_id,
answers)` keeps answers while the run still goes. `gated.resume(run_id, answers)` goes on
from the pause. Each answer is checked against its question, and a wrong one is refused
with every problem listed.

A person answers in one of four forms:

| Form | Shows | Answers |
|---|---|---|
| `ask` | `options`: a list of labels, or the path of a list the node reads | `{option}` |
| `judge` | `pair`: two values the node reads | `{pick}`: one of the two names, or `"tie"` |
| `approve` | `labels`: the yes and no choices | `{approved, option}` |
| `edit` | `edits`: the value to change | `{changed}`, and the new value under `writes={"answer": ..., "value": ...}` |

Every answer also holds `who`, `asked_at`, `answered_at`, `comment` and `timed_out`. The
question and the labels may name the node's reads, paths too, such as `{plan.title}`. The
visit keeps a record of kind `human`: what was shown, the answer as `raw_response`, a cost
of 0 and the wait in `wait_s`. An edited value must have the type of the value it
replaces, and a host's `answered_at` is an ISO time with its zone.

A pause outlives the process when the run store is a file: `MAS(..., store="runs.db")`,
or the variable `XMAS_RUNS`. A new process builds the same system and calls `resume`. A
changed system is refused. The budget counts on from the pause, and the wait is not wall
time. By default the store is the file of the [run log](reading-a-run.md#the-run-log). With
`XMAS_RUNS=:memory:` it is in memory, and a run resumes in the same process only.

`timeout_s` gives a question a deadline, and `default` is the answer then taken, with
`timed_out` true. It routes like any other answer. A host passes `asker=`: an object
whose `ask(question)` answers at once or returns `None`. It may also have
`withdraw(question, reason)`, called when a paused run is stopped. `run`, `run_batch` and
the optimizers never wait: a question the asker leaves open takes its default, or the run
fails and names the node, from inside a chain item or a box too. `ReplayAsker(traces)` answers from recorded runs, so an
optimizer can rerun a system with people in it.

## Stop

`mas.stop(run_id)` ends a running run from another thread. No call, step or item starts
after it, and a model call in flight is cut. Here Stop is pressed one second into a long
essay:

```python
import threading

essay = MAS(
    [Agent("writer", prompt="Write a long essay on: {query}", output="draft")],
    outputs={"draft": "text"},
    llm=llm,
)
threading.Timer(1.0, essay.stop, args=("essay-1",)).start()
trace = essay.run_state({"query": "Why is the sky blue?"}, run_id="essay-1")
print(trace.status, trace.error)  # stopped Stopped: stopped
```

The trace comes back within moments of the Stop and keeps every record so far. A visit
cut after a call keeps a failed record with what its calls used. No `on error` edge
catches a stop. A paused run is marked stopped, and its open questions are withdrawn,
as are those of a run stopped while it runs.

A Python step, a promptolution batch call or an MCP tool call is not cut. The run ends
at the next boundary after it, and a step may poll `xmas.cancel_flag()` to end early.
Streaming is opt-in per model: `APILLM(..., stream=True)` streams each answer, so a cut
call keeps the text received so far as its `raw_response`. Without it, `APILLM` sends the
same request as before.

## Racing branches: `first of`

A `first of` race starts several branches from one node, and the first good result
wins. Here a quick and a careful writer race to write the draft from one outline:

```python
outline = Agent("outline", prompt="Outline an answer to: {query}", output="outline")
quick = Agent("quick", prompt="One line from: {outline}", inputs=["outline"], output="draft")
careful = Agent("careful", prompt="A careful answer from: {outline}", inputs=["outline"], output="draft")
racing = MAS(
    [outline, quick, careful],
    edges=[Edge("outline", "quick", kind=EdgeKind.FIRST), Edge("outline", "careful", kind=EdgeKind.FIRST)],
    llm=llm,
)

trace = racing.run("Why is the sky blue?")
print(sorted(record.agent for record in trace.records if record.lost))  # ['careful']
print(trace.output)  # the quick draft
```

All `first of` edges of one node form one race. The winner's writes stand. The other
calls are cut, and each loser keeps a record with `lost` set and what it paid. A step
that races cannot be cut: it runs to its end, and its writes are dropped. Contestants
write the same values and go on to the same node, which runs once, or all end the run.
A box with a person inside does not race: a contestant never waits for an answer.
If every contestant fails, the race fails with the last failure. Inside a `for each`
chain, each item races on its own.
