# 🎄 xmas

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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

One file holds the whole system: prompts, wiring and models, never keys or endpoints.

```python
mas.to_yaml("team.yaml")
team = MAS.from_yaml("team.yaml", llm=llm)
```

## 🧠 Optimization built in

CANTANTE improves every agent's prompt at once, splitting each score into credit per agent.

```python
from promptolution.optimizers import OPRO
from xmas import GSM8KTask
from xmas.optimization import CantanteOptimizer, ContrastiveAttributor

task = GSM8KTask.from_huggingface(split="train", n=50)  # needs: uv run --with datasets
optimizer = CantanteOptimizer(mas, task, ContrastiveAttributor(llm), OPRO, optimizer_kwargs={"meta_llm": llm})
best = optimizer.optimize(n_steps=5)
best.to_yaml("team-optimized.yaml")
```

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
