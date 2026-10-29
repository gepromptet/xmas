# Bundles

## 💾 Save and share

A system saves as a bundle: a folder named `name.mas/`, or the same files zipped into one
file named `name.mas`. Here the system of [Conditions and loops](conditions-and-loops.md) goes into `team.mas` and
comes back:

```python
mas.save("team.mas")  # save("team.mas", zipped=True) writes one zip file
team = MAS.load("team.mas")
print(team.describe()["id"] == mas.describe()["id"])  # True
```

`MAS.load` builds each model the bundle names on the endpoint of your configuration (see
[Configuration](#configuration)). It finds each function where `callables.yaml` says it lives. Pass
`llm=`, `llms=` or `tools=[...]` to supply your own instead. The folder holds one file
per part:

```text
team.mas/
  xmas.yaml        # the wiring: inputs, outputs, nodes, edges, models by name
  callables.yaml   # where each function and tool lives, with typed reads and writes
  prompts/         # one plain-text file per prompt: writer.md, judge.md
  modules/           # each module's system, as a bundle of its own
  examples/        # warm-up cases for verify(), optional
  optimize.yaml    # what an optimizer may tune, optional
  xmas.lock        # checksums of every file and of each function's source
```

`xmas.yaml` is the system's spec without its prompts. No prompt text sits in the wiring:

```yaml
version: 2
agents:
  - {name: writer, reads: [query, verdict], writes: {draft: text}}
  - {name: judge, reads: [query, draft, count_words], writes: {verdict: json}}
code_nodes:
  - {name: count, uses: count_words}
  - {name: publish, uses: publish}
  - {name: sorry, uses: apologize}
edges:
  - writer -> count -> judge
  - judge -> writer: {when: not verdict.fits, max_visits: 3}
  - judge -> publish: {switch: verdict.action}
  - judge -> sorry: else
  - count -> sorry: {on error: [timeout, outage]}
default_model: llama-3.1-8b
```

A prompt file is the text exactly as the model sees it. A short header may hold the
answer's schema, and nothing else; xmas strips it before the model sees the text.
`prompts/judge.md`:

```markdown
---
schema:
  fits: {type: bool, means: the draft answers the question in under 50 words}
  action: {type: text, one of: [publish, shorten]}
---
Question: {query}
Draft, {count_words} words: {draft}
Judge the draft.
```

`callables.yaml` locates code and never inlines it. A remote tool names its server, here
`web`, and the server's address sits in your configuration under `[tools.web]`:

```yaml
count_words:
  kind: python
  import: mytools.text:count_words
  reads: {draft: text}
  writes: {count_words: json}
web_search:
  kind: mcp
  server: web
  writes: {web_search: text}
```

A python entry may add `params:`, constant arguments bound when the bundle loads. A model
name may stand for another catalogue id and carry parameters, under `models` in
`xmas.yaml`:

```yaml
models:
  editor: {model: llama-3.1-70b, max_tokens: 2000, timeout_s: 120, temperature: 0.2}
```

The parameters are `max_tokens`, `timeout_s`, `temperature`, `top_p`, `seed` and
`reasoning_effort`. They apply to the models `MAS.load` builds.

`xmas.lock` records the checksum of every other file and of each function's source, for
`verify()` to compare. Saving again rewrites only the files whose content changed. A new
prompt from an optimizer changes one prompt file and the lock, and your comments in the
other files stay. A bundle never holds an endpoint, a key, a tool server's address, a
budget or a run. A bundle may name any code, so by default `MAS.load` imports nothing: a
function must be a `@tool` in a module your program has already imported, or be passed
in `tools=[...]`. `MAS.load(path, trust=True)` imports the modules `callables.yaml` names
and wraps plain functions; use it only for bundles you trust.

A version 1 system, such as the [quickstart](https://github.com/gepromptet/xmas#readme)'s, also saves as one YAML file:

```python
team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)  # the quickstart's system
team.to_yaml("team.yaml")
team = MAS.from_yaml("team.yaml", llm=llm)
```

Any system's spec is plain data. `to_dict()` gives it, and `MAS.from_dict` rebuilds the
system with the same models and functions:

```python
spec = mas.to_dict()  # version, agents, code nodes, edges, callables, prompts
rebuilt = MAS.from_dict(spec, llm=llm, tools=[count_words, publish, apologize])
print(rebuilt.describe()["id"] == mas.describe()["id"])  # True
```

The spec's `edges` read like this in YAML, and `from_dict` accepts them in this form:

```yaml
- writer -> count -> judge
- judge -> writer: {when: not verdict.fits, max_visits: 3}
- judge -> publish: {switch: verdict.action}
- judge -> sorry: else
- count -> sorry: {on error: [timeout, outage]}
```

A whole spec in a YAML file loads with `MAS.from_yaml(path, llm=llm, tools=[...])`.
`to_yaml` refuses a version 2 system and points to `save`. Every version 1 file loads,
runs and saves as before.

## Verify

`verify()` checks a system on three levels, each including the ones before it:

- **0, checksums:** every file and each function's source is unchanged since it was
  saved, and the system in memory matches the files.
- **1, dry:** the wiring check, every prompt rendered, every answer form compiled, every
  examples file read, and every model name looked up in the endpoint's catalogue. No
  model is called. This is the default level.
- **2, warm-up:** the examples run under a small budget. A code node must write exactly what
  was recorded, and an agent's answer must pass its checks.

`save_examples` records one case per node from a run. Here the loaded team runs once and
keeps cases for the count and the judge, then runs them:

```python
trace = team.run("Why is the sky blue?")
team.save_examples(trace, nodes=["count", "judge"])
print(team.verify(2))
```

```text
level 0: unchanged since saved (bundle 9c7e2c30958e1c1d, 5 files)
level 1: no problems; model names checked against the endpoint catalogue
level 2: count  pass  0.0000 USD  <0.1 s
level 2: judge  pass  0.0100 USD  0.9 s
ok: 2 examples passed, 0.0100 USD
```

The cases land in `examples/warmup.yaml`, and the lock is refreshed. You can also write
cases by hand. A node case gives the values the node reads; a run case gives the system's
inputs and checks per output or node under `expect`:

```yaml
version: 1
budget: {cost_usd: 0.05, per example: {cost_usd: 0.03, wall_s: 120}}
examples:
- name: judge
  node: judge
  inputs: {query: "Why is the sky blue?", draft: Sunlight scatters off air., count_words: 4}
  recorded: {fits: true, action: publish}
  must: [{form: valid}, {path: verdict.fits, equals: true}]
- name: short-answer
  inputs: {query: "Why is the sky blue?"}
  expect: {judge: [{path: verdict.action, equals: publish}]}
```

A check is one of `form: valid`, `equals`, `contains`, `not contains`, `matches`, `line`,
`len`, `not empty`, `exists`, or `near` with `within`. Its `path` picks a value and a
field, such as `verdict.fits` or `pages[0]`. A picture input is a reference to a file
under `examples/media/`.

Without a budget, level 2 spends at most 0.05 USD in all and 0.02 USD per example. The
file's `budget` or `verify(2, budget=0.10)` sets another. A model whose cost is unknown
and that has no price fails its example, since no cap could bind. A report that is not
ok gives one `VerifyWarning`; `verify(strict=True)` raises `VerifyFailed` with the report
instead. `MAS.load(path, verify=1)` verifies while it loads, and `verify(2, only=["judge"])`
runs only that node's cases. Run it after a model switch, after an optimizer run and
before a release.

## The command line

The `xmas` command works on bundles from the shell:

```bash
xmas init team.mas         # a one-agent bundle to edit; --agents a,b for a chain
xmas check team.mas        # load and check, no model call
xmas show team.mas         # nodes and edges as text; --json for describe()
xmas run team.mas --query "Why is the sky blue?" --budget 0.10
xmas diff team.mas copy.mas  # file by file; with one bundle, against its lock
xmas verify team.mas --level 2
```

`run` takes other inputs as `--input name=value`: JSON for a json value, a path for a
picture. It prints the output, then one line with the status, the model calls, the cost
and the time. It refuses a model name the catalogue does not hold. `show`, `diff` and
`verify` take `--json`, and `verify --strict` fails when the report is not ok. The exit
status is 0 when all is well, 1 for a problem, and 2 for a usage error or a missing
bundle. `check`, `show`, `run` and `verify` take `--trust` to import the functions a bundle
names, from the current folder too.

`xmas runs` and `xmas trace <run id>` read the run log (see
[The run log](reading-a-run.md#the-run-log)).

## Configuration

A bundle names its models and tool servers. Where they are served lives in your
configuration: the environment, then a TOML file, `$XMAS_CONFIG` or else
`~/.config/xmas/config.toml` (under `$XDG_CONFIG_HOME` when it is set). A missing file is
no error. What you pass in code wins over both, such as
`MAS.load(path, config=Config.load(base_url=...))`. A file with every table:

```toml
[endpoint]
base_url = "http://localhost:8000/v1"
api_key_env = "GATEWAY_KEY"        # or api_key = "..."
catalogue = ["llama-3.1-8b"]       # optional; else GET {base_url}/models
stream = false

[models."llama-3.1-8b"]
price = {input_per_m = 0.6, output_per_m = 2.5}
stream = true

[tools.web]
url = "https://tools.example.com/mcp"
key_env = "WEB_TOOLS_KEY"          # or key = "..."

[store]
media = "~/.cache/xmas/store"
runs = "~/.cache/xmas/runs.sqlite"
```

Prices and streaming hold per model id, for the models `MAS.load` builds. A tool server
is named by `server:` in `callables.yaml` and found under `[tools.<name>]`. The
environment variables:

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |
| `XMAS_MODELS` | Model ids, comma separated, that stand for the catalogue, so names check offline |
| `XMAS_CONFIG` | The configuration file, instead of `~/.config/xmas/config.toml` |
| `XMAS_TOOLS_KEY` | Key for a tool server whose entry has none |
| `XMAS_STORE` | Folder for stored pictures and other media; default `~/.cache/xmas/store` |
| `XMAS_RUNS` | File of the run log, where every run is kept and paused runs resume after a restart; default `~/.local/share/xmas/runs.sqlite`, `:memory:` keeps runs in memory only |

The catalogue is the list of model ids the endpoint serves. It is `catalogue` or
`XMAS_MODELS` when set, else the endpoint's answer to `GET /models`, kept for five
minutes per endpoint and key. Offline, a kept answer of any age stands in. A model name
the catalogue does not hold gives a `ModelNameWarning` with the closest ids, such as
`llama-3.1-8b` for `llama-3-1-8b`. xmas never swaps in another name: the system refuses
to run until the name is fixed, and `verify(1)` checks again. Without an endpoint or a
catalogue, names go unchecked.

No key appears in a bundle, a record, a `repr` or the kept catalogue. A configuration
file that holds a key and that others can read gives a warning.
