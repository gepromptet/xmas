# 🎄 xmas

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/Coverage-97%25-brightgreen)
![Code Style](https://img.shields.io/badge/Code%20Style-ruff-black)
![Python Versions](https://img.shields.io/badge/Python%20Versions-3.12-blue)

🎁 **xmas builds, runs and optimizes multi-agent systems.** A system is a set of agents,
the plain code steps between them, the edges that wire them together and the model behind
each agent. xmas treats all of it as one object whose parts are ordinary parameters, so an
optimizer can improve every agent's prompt, the wiring and the choice of model against a
task you care about. Every run leaves a full per-agent trace with cost, time and errors,
so you always know who was naughty and who was nice. A single agent is the smallest
system, not a special case.

## 🚀 What is xmas?

In version 2, a system is a typed graph. Its nodes are agents, plain Python steps, people
who approve or edit, and boxes that hold a whole system. Its edges carry conditions, loops
and races, and a chain runs part of the graph once per item of a list. Every value has a
type: text, JSON, a picture or a file. xmas checks the wiring before the first model call
and draws the picture from the graph that runs. A system saves as a `.mas` bundle, a
folder of plain files with one file per prompt, and checks itself with a few warm-up
examples. Every run is logged, with its records, cost, time and errors. A system made only
of agents and plain edges is a version 1 system, and works as before.

xmas builds on [promptolution](https://github.com/gepromptet/promptolution): a system is a
promptolution predictor and a task is a promptolution task, so promptolution's optimizers
run on any agent unchanged. See [docs/design-principles.md](docs/design-principles.md) for
the principles behind the design.

### Key Features

- **Your first system in minutes:** a few `Agent` objects and their edges, and it runs.
- **One typed graph:** agents, code steps, people and boxes, with every value typed and the wiring checked before any call.
- **Conditions and loops:** `when`, `else`, `while`, `switch`, `on error` and `first of` edges, and `for each` chains.
- **Answers with a shape:** JSON answers with fixed fields, and one retry when an answer does not fit.
- **Pictures and files as values:** a model that can see gets the pictures, and the check refuses one that cannot.
- **People in the graph:** a person approves, picks, chooses or edits; the run pauses, and resumes after a restart too.
- **Every run logged:** one record per visit with time, tokens and cost, budgets that end a run cleanly, and health signals.
- **Save, share and optimize:** a `.mas` bundle checked by `verify()` and the `xmas` command, and optimizers for every prompt, the wiring and the models.

> **Status: beta.** Everything in this README is built. Learning from human picks is
> planned and not in the code yet (see [Where this is going](docs/guide/roadmap.md)).

## 📚 Scientific Publications

Methods implemented in xmas:

- **CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — Zehle, 2026. [arXiv](https://arxiv.org/abs/2605.13295) · `CantanteOptimizer`, `ContrastiveAttributor`

xmas builds on:

- **promptolution: A Unified, Modular Framework for Prompt Optimization** — Zehle et al., 2026. [EACL 2026](https://aclanthology.org/2026.eacl-demo.21/)
- **CAPO: Cost-Aware Prompt Optimization** — Zehle et al., 2025. [AutoML 2025](https://proceedings.mlr.press/v293/zehle25a.html)

## 🔧 Installation and Quickstart

xmas is not on PyPI yet. To use it in your own project:

```bash
uv add git+https://github.com/gepromptet/xmas
```

To work on xmas itself, clone it:

```bash
git clone https://github.com/gepromptet/xmas.git
cd xmas
uv sync
cp .env.example .env   # then set XMAS_BASE_URL and XMAS_API_KEY
```

xmas reads the two variables from the environment and never opens `.env` itself.
Export them, or start Python with `uv run --env-file .env python`. A configuration file
works too (see [Configuration](#configuration)). Installing xmas also installs the `xmas` command.

Tools served over MCP need `uv sync --extra mcp`.

### Quickstart: your first system

Two agents: an intern drafts an answer, and the boss turns it into the final one.
The question you pass in is the value `query`, which an agent reads unless `inputs=`
says otherwise. `{query}` and `{draft}` mark where each value goes in a prompt. The
intern's `output`, `draft`, is what the boss receives.

```python
from xmas import MAS, APILLM, Agent

llm = APILLM(model="llama-3.1-8b")  # endpoint and key come from XMAS_BASE_URL / XMAS_API_KEY

intern = Agent("intern", prompt="Answer this question, showing your working: {query}", output="draft")
boss = Agent(
    "boss",
    prompt="Question: {query}\nYour intern's draft: {draft}\nGive the final answer in one sentence, as your own.",
    inputs=["query", "draft"],
    output="prediction",
)

mas = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)

print(mas("Why is the sky blue?"))  # the boss's answer
print(mas.last_trace["intern"][0].outputs)  # {'draft': ...}: who really did the work
```

The model name must be one your endpoint serves. Each agent can run on its own model:
give it `model="..."` and pass the models by name through `MAS(..., llms={...})`.

### Version 2: a step and a JSON judge

A writer drafts, a plain Python step counts the words, and a judge answers in JSON with a
fixed field. The step makes no model call. `llm` is the quickstart's model.

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
    prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
    inputs=["query", "draft", "count_words"],
    output="verdict",
    output_type="json",
    schema={"fits": {"type": "bool", "means": "the draft answers the question in under 50 words"}},
)
mas = MAS(
    [writer, judge],
    steps=[Step("count", uses="count_words")],
    edges=[("writer", "count"), ("count", "judge")],
    tools=[count_words],
    llm=llm,
)

trace = mas.run("Why is the sky blue?")
print(trace.output)  # {"fits": true}
print([(record.agent, record.kind) for record in trace.records])
# [('writer', 'agent'), ('count', 'step'), ('judge', 'agent')]
```

The judge's model sees the schema as a few format lines after the prompt. An answer that
does not fit gets one retry, and the trace keeps every visit with its prompt, output, cost
and time.

### Save, load and verify

The same system saves as a `.mas` bundle and comes back from it:

```python
mas.save("team.mas")
team = MAS.load("team.mas")  # models from your configuration, functions already imported
print(team.describe()["id"] == mas.describe()["id"])  # True
print(team.verify())  # checksums and a dry check; no model is called
```

```text
team.mas/
  xmas.yaml        # the wiring: nodes, edges, models by name
  callables.yaml   # where count_words lives
  prompts/         # writer.md and judge.md, one plain-text file per prompt
  xmas.lock        # checksums of every file and of each function's source
```

A bundle never holds an endpoint, a key or a run. `team.verify(2)` also runs warm-up
examples under a small budget. The `xmas` command works on bundles from the shell, such
as `xmas check team.mas`, `xmas run team.mas --query "..."` and `xmas verify team.mas --level 2`.

## A tour of version 2

Each part has a page in the guide, with code that runs as written:

- [Steps and values](docs/guide/steps-and-values.md): plain Python between agents, and every wiring mistake listed before any call.
- [Answers with a shape](docs/guide/answer-forms.md): JSON answers with fixed fields, and one retry.
- [Conditions and loops](docs/guide/conditions-and-loops.md): `when`, `else`, `while`, `switch` and `on error` edges, joins and merges.
- [Reading a run](docs/guide/reading-a-run.md): records, health signals, budgets, and the run log.
- [Drawing a system](docs/guide/drawing-a-system.md): `describe()`, the graph that runs as plain data.
- [A system inside a system](docs/guide/boxes.md): a box runs a whole system as one node.
- [Once per item](docs/guide/chains.md): a `for each` chain runs some nodes once per item of a list.
- [Several values in, several out](docs/guide/run-state.md): `run_state`, events as the run goes, and retries.
- [Pictures, documents and other files](docs/guide/media.md): media values by reference, checked against what each model accepts.
- [People, Stop and races](docs/guide/people.md): a person in the graph, a Stop that stops, and `first of` races.
- [Bundles](docs/guide/bundles.md): save, load, the lock, `verify()`, warm-up examples, the `xmas` command and configuration in full.
- [Optimization](docs/guide/optimization.md): CANTANTE, frozen nodes, topology search, and xmas with promptolution.
- Planned: [learning from human picks](docs/guide/roadmap.md).

## 🏗 Components

- **`Agent`** – One node: a prompt template, the values it reads and the value it writes.
- **`Step`** – A node that runs a function or an MCP tool from the system's `tools=`, with no model call.
- **`Edge`** – One edge with its operator, an `EdgeKind`: plain, `when`, `else`, `while`, `switch`, `on error` or `first of`.
- **`Box`** – A whole system used as one node of another.
- **`Human`** – A node that asks a person, in one of four forms: `ask`, `judge`, `approve` or `edit`.
- **`MAS`** – The system and the predictor: agents, steps, edges, models and tools; runs a query and keeps its trace.
- **`Trace`** – The per-agent record of one run: rendered prompts, outputs, tokens, cost, time, tool calls, status.
- **`ForEach`** – The `for each` form of a chain: the item, the list it comes from, and options such as `at_once` and `key`.
- **`Ref`** – A picture, a sound, a document, a table or a file as a value: where it is, its type, size and checksum.
- **`Capabilities`** – What a model accepts and produces beyond text, such as `Capabilities(accepts=("image",))`.
- **`Event`** – One start or end of a visit or an item, a pause, a resume or a stop, passed to `on_event` as the run goes.
- **`RunStore`** – Where paused runs, their questions and answers are kept: in memory, or in one SQLite file.
- **`Budget`** – Limits on calls, cost and wall time for each run, and on model calls in flight.
- **`Config`** – Where models and tools are served: endpoint, key, catalogue, prices and tool servers.
- **`VerifyReport`** – What `verify()` found on each level, with each example's result, cost and time.
- **`Health`** – Signals read from a finished run: stuck loops, empty answers, retries, caps hit.
- **`MASTask`** – A dataset plus a reward over the system's answers, e.g. `GSM8KTask`.
- **`APILLM`** – Any OpenAI-compatible endpoint, with scoring, tool turns and a per-call log.
- **`MASOptimizer`** – Propose, evaluate through the task, select; CANTANTE and topology search build on it.

## Configuration

A bundle names its models and tool servers. Where they are served lives in your
configuration: the environment, then a TOML file, `$XMAS_CONFIG` or else
`~/.config/xmas/config.toml`. A bundle never holds an endpoint or a key.

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |
| `XMAS_RUNS` | File where every run is logged; default `~/.local/share/xmas/runs.sqlite`, `:memory:` opts out |
| `XMAS_STORE` | Folder for stored pictures and other media; default `~/.cache/xmas/store` |

`xmas runs` lists the logged runs, and `xmas trace <run id>` shows one. The other
variables, the file's tables, model prices and tool servers are in
[Configuration](docs/guide/bundles.md#configuration).

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, the
commit convention and review expectations, and [CHANGES.md](CHANGES.md) for what changed
in version 2. The checks CI runs:

```bash
uv sync                                    # create .venv and install everything
uv run ruff check . && uv run ruff format --check .
uv run mypy                                # strict type check
uv run pytest                              # tests, the guide's code blocks among them
uv run mkdocs build --strict               # docs
```

## 📄 Citation

If you use xmas in your research, please cite:

```bibtex
@software{zehle2026xmas,
  title={xmas},
  author={Zehle, Tom},
  year={2026},
  url={https://github.com/gepromptet/xmas}
}
```

## License

See [LICENSE](LICENSE).
