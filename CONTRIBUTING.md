# Contributing to xmas

Thanks for working on xmas. This document is short on purpose: the tooling enforces most
of the rules, so there is little to memorize.

## Setup

```bash
uv sync
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

`uv sync` creates `.venv` and installs the project plus the tooling: `dev` is a
PEP 735 dependency group, and uv installs it by default.
Both hook types matter: `pre-commit` formats and checks the diff, `commit-msg` validates
the commit message.

## Workflow

`main` is protected and always releasable. All work lands through pull requests.

1. Branch from `main`. Name it `<type>/<short-description>`, for example
   `feat/graph-mutation` or `fix/router-retry-leak`.
2. Commit using [Conventional Commits](https://www.conventionalcommits.org/) (see below).
3. Open a PR. CI runs lint, types and tests; all three must be green.
4. Get a review, then squash-merge. The PR title becomes the commit message on `main`,
   so it must also follow the commit convention.

Keep pull requests small enough to review in one sitting. A PR that changes behaviour
comes with a test that would fail without it.

## Commit messages

Enforced by `conventional-pre-commit`:

```
<type>(<optional scope>): <description>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`, `build`.

```
feat(graph): add topology mutation operators
fix(router): reset retry counter when a task completes
docs: document the XMAS_BASE_URL convention
```

## Code standards

| Concern | Tool | Command |
|---|---|---|
| Lint | ruff | `uv run ruff check .` |
| Format | ruff | `uv run ruff format .` |
| Types | mypy (strict) | `uv run mypy` |
| Tests | pytest | `uv run pytest` |

House rules the tools do not catch:

- **Public API is typed and documented.** Google-style docstrings, checked by ruff's `D`
  rules. Strict mypy means no untyped defs and no implicit `Any`.
- **Line length is 120.**
- **No credentials, ever.** Configuration comes from the environment
  (`XMAS_BASE_URL`, `XMAS_API_KEY`). `.env` is gitignored and gitleaks scans every commit.
- **No run artifacts.** Results, logs, checkpoints and datasets stay out of the repo.
- **Tests do not require a network by default.** Anything that needs the LLM gateway
  skips itself when `XMAS_BASE_URL` and `XMAS_API_KEY` are unset.

## Dependencies

Add runtime dependencies with `uv add <pkg>`, tooling with
`uv add --dev <pkg>`, and backend-specific ones with
`uv add --optional <extra> <pkg>`. All of them update `pyproject.toml`. `uv.lock` is
not committed: it stays local, and CI resolves from `pyproject.toml`.

Keep the runtime dependency list short. `promptolution` is a deliberate hard dependency
(xmas subclasses its base classes); anything heavy goes behind an extra.

Note that promptolution 2.2.3 declares `>=3.10,<3.13`, so xmas is pinned to Python 3.12
until promptolution widens its supported range.

## Versioning

`__version__` lives in `src/xmas/__init__.py` and the build reads it from there. Keep it a
plain string literal on its own line. Do not duplicate the version into `pyproject.toml`.

## Docs

Docs are mkdocs-material with mkdocstrings, sourced from docstrings. `uv run mkdocs serve`
previews locally; CI builds them with `--strict`, so a broken link or a bad reference
fails the build.
