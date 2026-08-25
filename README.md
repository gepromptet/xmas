# xmas

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**xmas is the multi-agent sibling of [promptolution](https://github.com/gepromptet/promptolution).**
It keeps the same five nouns, Task, Predictor, LLM, Optimizer and ExperimentConfig, and
lifts them one level: the multi-agent system itself is the Predictor, a structured one
whose parameters are exposed as ordinary optimizable components, so per-agent prompts,
the topology wiring the agents together, and the model bound to each node are all things
an optimizer can improve against a task you care about. Because xmas builds directly on
promptolution's base classes rather than reimplementing them, a single-agent system is
not an analogy to a promptolution predictor, it is exactly one, and promptolution's
optimizers plug in without a port. Within the gepromptet ecosystem that makes
promptolution the N=1 case and xmas the general one.

See [docs/design-principles.md](docs/design-principles.md) for the six principles that
hold the design together.

> **Status: pre-alpha.** Core inference and the optimization loop are in place.
> Attribution methods, topology search and the experiment layer are not.

## Installation

xmas is not published to a package index. Install from source:

```bash
git clone https://github.com/gepromptet/xmas.git
cd xmas
uv sync
```

The core depends on `promptolution` and little else. Heavy backends live behind
extras, mirroring promptolution's own `[api]` / `[vllm]` / `[transformers]` split:

```bash
uv sync --extra vllm     # vLLM backend
```

## Quickstart

```python
from xmas import Agent, MAS, APILLM

llm = APILLM(model="llama-3.1-8b", base_url="https://api.tomzehle.com/v1")

solver  = Agent("solver",  prompt="Solve step by step.", output="draft")
checker = Agent("checker", prompt="Verify and answer.", inputs=["query", "draft"], output="prediction")

mas = MAS([solver, checker], edges=[("solver", "checker")], llm=llm)

answer = mas("What is 17 * 23?")          # __call__ runs it
trace  = mas.last_trace                    # full per-agent record of that call

mas.to_yaml("gsm8k.yaml")                  # serialization as methods
mas2 = MAS.from_yaml("gsm8k.yaml", llm=llm)
new = mas.copy(prompts={"solver": "..."})  # optimizers derive candidates this way
```

Agents are addressable and mutable through the system:

```python
mas.agents["solver"].prompt = "Solve carefully, showing each step."
```

## The degenerate case

A one-agent system is a promptolution predictor, not an analogue of one, so
promptolution's own optimizers run against it with no adapter:

```python
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.prompt import Prompt

mas = MAS([Agent("solver", prompt="Classify: {query}", output="prediction")], llm=llm)
task = ClassificationTask(df, x_column="x", y_column="y")

result = task.evaluate(Prompt("Classify: {query}"), predictor=mas)
```

## Optimizing

Every part of a system is a component an optimizer can change. The loop is
propose, evaluate through the task, select:

```python
from xmas.optimization import PrefixSearchOptimizer
from xmas.tasks.gsm8k import GSM8KTask

task = GSM8KTask.from_huggingface(split="test", n=50)
best = PrefixSearchOptimizer(mas, task).optimize(n_steps=3)
```

`PrefixSearchOptimizer` is deliberately trivial and exists as a control: it
exercises the whole loop with no algorithm in the way. Real methods arrive as
`Attributor` and `Updater` implementations, not as new frameworks.

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
uv sync --extra dev                        # create .venv and install everything
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy                               # strict type check
uv run mkdocs serve                       # docs at http://127.0.0.1:8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, commit convention and review
expectations.

## License

MIT. See [LICENSE](LICENSE).
