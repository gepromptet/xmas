# 🎄 xmas

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🚀 What is xmas?

🎁 **xmas builds, runs and optimizes multi-agent systems.** A system is a set of agents,
the edges that wire them together and the model behind each agent. xmas treats all of it
as one object whose parts are ordinary parameters, so an optimizer can improve every
agent's prompt, the wiring and the choice of model against a task you care about. Every
run leaves a full per-agent trace, so you always know who was naughty and who was nice.

### Key Features

- **Your first system in minutes:** a few `Agent` objects and their edges, and it runs.
- **Save and share as YAML:** a whole system (prompts, wiring, models) in one file anyone can load.
- **Optimization built in:** improve every agent's prompt, the wiring and the models against your task.

See [docs/design-principles.md](docs/design-principles.md) for the principles behind the
design.

> **Status: beta.** Inference, the optimization loop, CANTANTE and topology search are in
> place; the experiment layer is still to come.

## 🔧 Installation

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

Tools served over MCP need `uv sync --extra mcp`.

## 💡 Your first system in minutes

Two agents: an intern drafts an answer, and the boss turns it into the final one.
`{query}` and `{draft}` name what each agent reads; the intern's `output` is what the boss
receives.

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
print(mas.last_trace["intern"][0].outputs)  # who really did the work
```

Each agent can run on its own model: give it `model="..."` and pass the models by name
through `MAS(..., llms={...})`.

## 💾 Save and share as YAML

A whole system is one YAML file: every prompt, the wiring and the model each agent runs
on. Endpoints and keys never go into it, so the file is safe to commit or send to a
colleague, who loads it against their own endpoint.

```python
mas.to_yaml("team.yaml")

team = MAS.from_yaml("team.yaml", llm=APILLM(model="llama-3.1-8b"))
print(team("Why is the sky blue?"))
```

`team.yaml`, exactly as written:

```yaml
version: 1
agents:
- name: intern
  prompt: 'Answer this question, showing your working: {query}'
  inputs:
  - query
  output: draft
  tools: []
  model: null
- name: boss
  prompt: 'Question: {query}

    Your intern''s draft: {draft}

    Give the final answer in one sentence, as your own.'
  inputs:
  - query
  - draft
  output: prediction
  tools: []
  model: null
edges:
- - __start__
  - intern
- - intern
  - boss
- - boss
  - __end__
default_model: llama-3.1-8b
```

## 🧠 Optimization built in

CANTANTE optimizes every agent's prompt at once. Each agent gets its own promptolution
optimizer; all of them share one system evaluation, and a meta-model compares runs of the
same question to split the score into credit per agent. An agent whose change helped sees
a better score, one whose change hurt sees a worse one.

```python
from promptolution.optimizers import EvoPromptGA

from xmas import MAS, APILLM, Agent, GSM8KTask
from xmas.optimization import CantanteOptimizer, ContrastiveAttributor

llm = APILLM(model="llama-3.1-8b")

mas = MAS(
    [
        Agent("solver", prompt="Solve: {query}\nShow your working.", output="draft"),
        Agent("checker", prompt="Reply with only the final number.\n\n{draft}", inputs=["draft"], output="prediction"),
    ],
    edges=[("solver", "checker")],
    llm=llm,
)

task = GSM8KTask.from_huggingface(split="train", n=50)  # needs datasets: uv run --with datasets python ...
optimizer = CantanteOptimizer(
    mas,
    task,
    attributor=ContrastiveAttributor(llm),  # credit per agent, from contrasting runs
    optimizer_cls=EvoPromptGA,  # any promptolution optimizer, one per agent
    initial_prompts={
        "solver": ["Solve: {query}\nShow your working.", "Work step by step.\n\n{query}"],
        "checker": ["Reply with only the final number.\n\n{draft}", "State just the number.\n\n{draft}"],
    },
    optimizer_kwargs={"meta_llm": llm},
)

best = optimizer.optimize(n_steps=5)
print(optimizer.history)  # task score of the best system after each step
print(optimizer.credit)  # mean credit per agent in the last round
best.to_yaml("gsm8k-best.yaml")
```

`EvolutionaryTopologyOptimizer` searches the wiring the same way, and
`PrefixSearchOptimizer` is a deliberately trivial control that runs the loop with no
algorithm in the way.

## 🔗 xmas and promptolution

xmas builds on [promptolution](https://github.com/gepromptet/promptolution): a system is a
promptolution predictor and a task is a promptolution task, so promptolution's optimizers
run on any agent unchanged.

## 📚 Scientific Publications

Methods implemented in xmas:

- **CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — Zehle, 2026. [arXiv](https://arxiv.org/abs/2605.13295) · `CantanteOptimizer`, `ContrastiveAttributor`

xmas builds on:

- **promptolution: A Unified, Modular Framework for Prompt Optimization** — Zehle et al., 2026. [EACL 2026](https://aclanthology.org/2026.eacl-demo.21/)
- **CAPO: Cost-Aware Prompt Optimization** — Zehle et al., 2025. [AutoML 2025](https://proceedings.mlr.press/v293/zehle25a.html)

## 🏗 Components

- **`Agent`** – One node: a prompt template, the values it reads and the value it writes.
- **`MAS`** – The system and the predictor: agents, edges, models and tools; runs a query and keeps its trace.
- **`Trace`** – The per-agent record of one run: rendered prompts, outputs, tokens, tool calls.
- **`MASTask`** – A dataset plus a reward over the system's answers, e.g. `GSM8KTask`.
- **`APILLM`** – Any OpenAI-compatible endpoint, with scoring, tool turns and a per-call log.
- **`MASOptimizer`** – Propose, evaluate through the task, select; CANTANTE and topology search build on it.

## Configuration

xmas is configured entirely through the environment. There is no config file, and no
credential is ever read from or written to the repository.

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |

Copy the template and fill it in locally:

```bash
cp .env.example .env
```

`.env` is gitignored, `.env.example` holds dummy values only, and a
[gitleaks](https://github.com/gitleaks/gitleaks) pre-commit hook scans every commit for
credentials. If you need a key in CI, add it as a repository secret and expose it as an
environment variable in the workflow. Never inline it.

Tests that need a live model run against the gateway at
[`api.tomzehle.com`](https://api.tomzehle.com), which speaks the OpenAI protocol. Point
`XMAS_BASE_URL` at it and set `XMAS_API_KEY` to your gateway key. Tests that require a
gateway are skipped when the variables are absent, so a fresh clone is green without any
credential.

## Development

```bash
uv sync                                    # create .venv and install everything
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy                               # strict type check
uv run mkdocs serve                       # docs at http://127.0.0.1:8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, commit convention and review
expectations.

## 📄 License

MIT. See [LICENSE](LICENSE).
