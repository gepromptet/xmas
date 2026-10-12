# 🎄 xmas: multi-agent systems, wrapped and ready 🎁

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🎅 What is xmas?

**xmas builds, runs and optimizes multi-agent systems.** Think of Santa's workshop 🎄: one
elf reads the wish list, another builds the toy, a third wraps it 🎁, and Santa checks the
list twice before anything goes down the chimney 🦌. That team is a multi-agent system,
and xmas treats the whole workshop as one object: every agent's prompt, the edges that
wire the agents together and the model behind each agent are ordinary parameters that an
optimizer can improve against a task you care about. Every run leaves a full per-agent
trace, so you can see which elf dropped the present. ❄️

### Key Features

- Build a system from plain `Agent` objects and edges; cycles are allowed and bounded.
- Optimize prompts with CANTANTE, which splits one system score into credit per agent.
- Search the topology itself: add, remove and rewire agents under a budget.
- Bind a different model to each agent, and record tokens and cost per call.
- Save and load a system as YAML; derive candidates with `copy` without touching the original.
- Give agents tools, as Python functions or from an MCP server.

See [docs/design-principles.md](docs/design-principles.md) for the principles behind the
design.

> **Status: pre-alpha.** Inference, the optimization loop, CANTANTE and topology search are
> in place; the experiment layer is not.

## 🔧 Installation

xmas is not published to a package index yet. Install from source:

```bash
git clone https://github.com/gepromptet/xmas.git
cd xmas
uv sync
cp .env.example .env   # then set XMAS_BASE_URL and XMAS_API_KEY
```

Tools served over MCP need `uv sync --extra mcp`.

## 🚀 Quickstart: Santa's workshop

Two agents: an elf proposes a gift, and Santa makes the final call. `{query}` and
`{gift}` name what each agent reads; the elf's `output` is what Santa receives.

```python
from xmas import MAS, APILLM, Agent

llm = APILLM(model="llama-3.1-8b")  # endpoint and key come from XMAS_BASE_URL / XMAS_API_KEY

elf = Agent("elf", prompt="You are a workshop elf. Suggest one gift for this wish list: {query}", output="gift")
santa = Agent(
    "santa",
    prompt="You are Santa. Wish list: {query}\nYour elf suggests: {gift}\n"
    "Give the final gift in one line. No ponies, the sleigh is full.",
    inputs=["query", "gift"],
    output="prediction",
)

workshop = MAS([elf, santa], edges=[("elf", "santa")], llm=llm)

print(workshop("A pony, a rocket and a PhD"))  # Santa's answer
print(workshop.last_trace["elf"][0].outputs)  # what the elf proposed, from the trace

workshop.to_yaml("workshop.yaml")  # save it, then MAS.from_yaml("workshop.yaml", llm=llm)
nicer = workshop.copy(prompts={"santa": workshop.agents["santa"].prompt + " Be jolly."})
```

Each agent can run on its own model: give it `model="..."` and pass the models by name
through `MAS(..., llms={...})`.

## 🧠 Optimizing a system

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

task = GSM8KTask.from_huggingface(split="train", n=50)  # needs `uv pip install datasets`
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
