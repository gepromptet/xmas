# API reference

Generated from docstrings. Everything on this page is public surface; anything
not on it is an implementation detail and may change without notice.

## Defining a system

::: xmas.agent.Agent

::: xmas.mas.MAS
    options:
      members:
        - compiled
        - nodes
        - structure
        - terminal
        - while_edges
        - __call__
        - __init__
        - __repr__
        - answer
        - capabilities
        - copy
        - describe
        - encode
        - from_dict
        - from_yaml
        - load
        - predict
        - prompts
        - resolve_callable
        - resolve_llm
        - resolve_tools
        - resume
        - run
        - run_batch
        - run_state
        - save
        - save_examples
        - stop
        - to_dict
        - to_yaml
        - verify

## Topology

The graph is a plain `networkx.DiGraph` over agent names plus two sentinels, so a
topology optimizer is an ordinary graph-rewriting component.

A cycle is expressed as an ordinary edge. The engine classifies the edge that
closes it as a back edge and compiles it to a conditional branch, so the agent
that closes the loop decides whether to take it by emitting `<continue>` in its
response. Two guards bound the loop regardless of what a prompt says: that
agent's own visit count, and `max_steps` behind it.

::: xmas.graph
    options:
      members:
        - START
        - END
        - build_graph
        - validate_graph
        - agent_names
        - back_edges
        - forward_predecessors
        - join_nodes
        - terminal_agent

## Traces

The record a run leaves behind, and the input an attributor computes on.

::: xmas.trace.Trace

::: xmas.trace.AgentRecord

::: xmas.trace.ToolCall

::: xmas.trace.TokenUsage

::: xmas.trace.Event

## Budgets and errors

`MAS(..., budget=Budget(...))` limits every run, and every item of a batch, on
its own. A call starts only while the run is below every limit, so a limit of
`0` spends nothing, and calls already in flight may overshoot a budget, by at
most one model call, with its retries, per parallel branch. A cost limit stops
the run before the next call when a call reported no cost and its model has no
price, because a limit that cannot be checked would not bind. A run stopped
this way is data, not an exception: its trace has status `over_budget`, and
every failed trace carries an `error_kind`.

On xmas's own model class, `APILLM`, permanent errors (a 4xx other than 408,
409 and 429, or a 4xx that names money) are not retried;
`APILLM(full_response=False)` restores promptolution's path for plain agents,
which retries every error.

::: xmas.budget.Budget

::: xmas.errors
    options:
      members:
        - ErrorKind
        - RunStatus
        - XmasError
        - BudgetExceeded
        - Stopped
        - RaceLost
        - Unanswered
        - EmptyResponse
        - BadFormat
        - classify
        - kind_of_status
        - kind_of_finish
        - status_for

## Models

Each backend subclasses its promptolution counterpart and adds `score()`, the
teacher-forced sibling of generation.

`APILLM(full_response=None)`, the default, sends plain agents' calls through
xmas's own path, which sends the same request and keeps the provider's tokens,
cost, finish reason and served model on the agent's record, as tool turns
already do; a subclass that overrides a hook of promptolution's path keeps that
path, and `full_response=False` keeps it for any model. `APILLM(price=TokenPrice(...))` prices the tokens of a call
whose response reports no cost; without a price, that cost stays unknown
(`None`), never `0`. `APILLM(accepts=("text", "image"), produces=("text",))`
states what the model takes and gives (see Version 2: media values); a
prompt with a picture or a sound goes out as content parts, and a prompt
without one keeps its exact request bytes.

::: xmas.llms.api_llm.APILLM
    options:
      members:
        - model
        - __init__
        - call_params
        - generate_image_with_meta
        - generate_parts_with_meta
        - generate_with_meta
        - get_response
        - get_tool_response
        - get_tool_response_meta
        - record
        - score

Per-call records: pass `call_log=CallLog()` and every call leaves a `CallRecord` with a
`call_id` (UUID4), a `ts` (start time, ISO 8601 with offset), the `params` it was sent
(`max_tokens` plus `call_kwargs`, minus `extra_headers`) and the provider's `response_id` (`response_ids` for a batch).
`CallLog(capture_text=True)` also records `prompt`, `system_prompt` and `response`; a batch
stays one record and fills `prompts`, `system_prompts`, `responses` instead.
A provider or tool error raises `XmasError` (under Budgets and errors), which
carries the HTTP `status` when there is one.

::: xmas.llms.scoring.ScoredSequence

::: xmas.llms.meta.TokenPrice

::: xmas.llms.meta.CallMeta

::: xmas.llms.meta
    options:
      members:
        - SupportsParts
        - SupportsImages
        - supports_parts
        - supports_images

## Tools

Functions an agent may call while it answers: `Agent.tools` names resolve
against `MAS(..., tools=[...])` when the system is built. `MCPTools` needs the
`mcp` extra. A tool turn sends sampling parameters (`temperature`, `top_p`,
`stop`, ...) only when they are set in the model's `call_kwargs`, never as
defaults. Every requested call runs, since a tool call need not be idempotent.

::: xmas.tools
    options:
      members:
        - Tool
        - tool
        - ToolSource
        - ToolCallRequest
        - LocalTools
        - run_code

::: xmas.tools.mcp.MCPTools

::: xmas.llms.masking
    options:
      members:
        - assistant_span
        - mask_logprobs

## Version 2: code nodes, typed values, answer forms

A system is version 2 when it uses anything below (or `MAS(..., version=2)`);
a system that uses none of it is version 1, with its own spec format. What a
version 2 system can do:

- **Code nodes.** `MAS(..., code_nodes=[Code("count", uses="count_words")])` runs a
  callable from `tools=` as a node, with no model call. The same callable can
  also be an agent's tool. A code node reads its parameters and writes one value
  named like the callable; `Code(reads={param: value}, writes={write: value})`
  renames them, and `tool(writes={"counts": "json"})` declares other writes.
- **Typed values.** Every value is `text` or `json` (`TEXT`, `JSON`), or a
  list of them at any depth (`list of text`, `list of list of json`); a
  callable's `string` parameters read text, the others json, and its write is
  text when the function returns `str` (or the tool is remote), else json. A
  json value or a list reaches a prompt as one line of JSON, and a list may be
  read where json is declared (`fits`). `check_value` checks every item of a
  list; an item may be `null`. Images, audio, documents, tables and files are
  values too, passed by reference (see Version 2: media values).
- **Answer forms.** `Agent(..., output_type="json", schema={...})` makes the
  agent answer in JSON. Its prompt ends with format lines no optimizer can
  drop; an answer that does not fit gets exactly one retry, then the run fails
  with `BadFormat` (kind `bad_format`). The record keeps the rejected answer.
- **Merges and joins.** A node fed by several edges is a merge when its edges
  never arrive in one pass (it runs on whichever arrives) or a join when it
  waits for one edge of every group (it runs once per pass); the check refuses
  any other node and says which edges clash.
- **Checked when built.** `check_system` lists every problem at once: a read
  nobody writes before the node runs, two writers of one value (unless
  `state={value: {"merge": "last"}}`), a placeholder that is not a read, a type
  that does not fit a parameter, a reserved name (the type words such as
  `text` and `json`, and `visits`, `cost`, `len`, ...), an unknown callable.

The answer-form dialect, one entry per field:
`{type, means?, optional?, one of?, fields?}`, where `type` is `text`, `bool`,
`int`, `number`, `json`, `object` or `list of <type>`, and `fields` describes
an object or a list's objects.

A version 2 spec (`version: 2`) has the keys `inputs`, `outputs`, `agents`
(`model`, `prompt`, `for each`, `reads`, `writes: {name: type}`, `tools`,
`schema`, `retry`), `code_nodes` (`uses`, `for each`, `reads`, `writes`, `retry`),
`modules` (`for each`, `reads`, `writes`, `system`), `edges` (a string is a
chain of edges, `a -> b -> c`), `state`, `health`, `models` (what
each named model accepts and produces), `callables` (what each callable reads
and writes), `default_model` and `prompts`. An agent's
`reads` is a list of values or a map `{placeholder: value}`; its `prompt` names
an entry of `prompts` that several agents may share, by default its own name.
Models
appear by name only and keys never appear: endpoints and keys come from
configuration, never from the spec.

Refused, with a message saying so: `to_yaml` on a version 2 system (`save(path)`
writes it as a `.mas` bundle; `to_dict()` gives its spec), and `run(query)` on a
system whose inputs are not exactly `{query: text}` (`run_state(values)` runs it).

::: xmas.nodes.Code

::: xmas.edges
    options:
      members:
        - Edge
        - EdgeKind

::: xmas.values
    options:
      members:
        - ValueType
        - TEXT
        - JSON
        - parse_type
        - LIST_PREFIX
        - check_value
        - fits
        - render_value
        - FILE
        - MEDIA
        - ref_kind
        - plain
        - coerce_ref

::: xmas.schemas
    options:
      members:
        - compile_form
        - AnswerForm
        - FormError
        - RETRY_NOTE

::: xmas.check
    options:
      members:
        - check_system
        - SystemInfo
        - RESERVED_READS
        - RETRYABLE

::: xmas.routes
    options:
      members:
        - Join

## Version 2: operators, modules, health

Edges decide where a run goes next. In a spec, a chain `a -> b` is plain (always
taken); a one-key mapping gives the operator:

- **`when`.** `judge -> publish: {when: verdict.fits}` is taken when the
  predicate holds, judged on the node's fresh writes.
- **`else`.** `judge -> apology: else` is taken when no `when`, `while` or
  `switch` edge of the same node is.
- **`while`.** `judge -> writer: {when: not verdict.fits, max_visits: 3}` loops
  back until its target ran `max_visits` times; then the record is `capped` and
  the `else` edge is taken, even when a later `when` edge holds. The count
  includes the run before the loop, so `max_visits: 2` is one redraw, and a
  cap of 1 on `check -> redraw` runs `redraw` once.
- **`switch`.** `judge -> shorten: {switch: verdict.action}` is taken when the
  path equals the case (the target's name unless `case:` says otherwise); a
  value outside the cases takes `else`.
- **`on error`.** `count -> apology: {on error: [timeout, outage]}` (or `any`)
  catches the node's failure of those kinds: the visit keeps a failed record and
  the run goes on. Budget and stop are never caught.

Predicates are a small language, never `eval`: paths (`verdict.fits`),
literals, `== != < <= > >= in`, `and or not`, `exists(path)`, `len(x)`,
`visits`, `cost`, `elapsed`, and registered callables that return a bool. The
check refuses an unknown name before a run.

A **module** (`MASModule(name, system, reads, writes)`, spec `modules:`) runs a whole
version 2 system as one node: `reads` maps its inputs from outer values,
`writes` its values back out. It spends from the outer budget, and its record
holds the inner trace (`Trace.walk()` yields every record with its module path).

::: xmas.nodes.MASModule

`detached: true` on a node (`Agent(..., detached=True)`) keeps it as it is through every
optimizer: no prompt, model, read, write or edge of it changes, none removes it.
The topology search also leaves conditional edges and their endpoints alone,
holds every member of a `for each` chain, keeps code nodes and modules, and drops a
candidate the check refuses.

Every run gets **health** signals in `Trace.health`, for information only:
`retried`, `bad_format`, `error_routed`, `item_failed`, `empty`, `stuck`,
`unfilled`, `leak`, `long`, `short` and `item_over_budget` per record, how often
each `while` edge was taken, each set budget limit's use, and per chain its
items, failed items and each item's cost (`Health.items`). A spec tunes them with `health: {length_factor,
min_samples, leak}`; a healthy run's trace dict has no `health` key.

`MAS.describe()` returns plain data: every node with its kind, model name,
reads, writes and `detached`; every edge with its kind and condition; the join of
each node fed by several edges; an `id` hashed from it. Like the spec, it names
models only: endpoints and keys come from configuration.

::: xmas.predicates
    options:
      members:
        - parse_predicate
        - Predicate
        - PredicateError

::: xmas.health
    options:
      members:
        - Health
        - HealthSettings
        - Flag
        - summarize

## Version 2: for each, run_state, events

A **`for each` chain** runs some nodes once per item of a list:
`MAS(..., each={"writer": ForEach("page", "plan.slides", at_once=4, key="id"),
"render": "page in plan.slides"})`, or `for each: page in plan.slides` on each
node of a spec. Every member names the same item and list; the options
`at_once` (items in flight), `key` (a scalar field that names an item in its
records) and `item_budget` (`cost_usd`, `wall_s`: flagged as
`item_over_budget`, never stopped) sit on the head, the node every edge from
outside lands on. Inside, the item has the list's item type (json for a path
with fields) and each write its declared type; outside, a write is a list in
item order. Exits leave the chain by one plain or `else` edge to one target;
a chain with no exit ends the run and needs declared outputs. Chains do not
nest, also not through a module.

Items run in parallel behind LangGraph's `Send`, each on a run of the chain's
own system with the item's index and key on every record (`AgentRecord.item`,
`item_key`, `visit`). An item that fails with any kind but `budget`, `stopped`
or an unrouted `money` gets a failed record and `None` in every gathered list,
and the run goes on; those three end the run. A `while` edge inside a chain
loops per item, and its cap counts per item; a `while` edge into the head
counts passes of the whole chain. `Budget(max_parallel_calls=n)` caps the model
calls in flight across branches, modules and items. Each graph run has a pool of up
to `xmas.runtime.BRANCH_THREADS` (64) threads, whatever the host's cores, so the
branches and `at once` items of a run truly run together on a small machine too.

**`MAS.run_state(values, on_event=None)`** runs a version 2 system on all of
its inputs by name (a version 1 system on text values, `query` among them). A
missing, extra or mistyped input raises one `ValueError` listing them all,
before any call. `MAS(..., outputs={"deck": "json", "notes": "text"})` names
the values the trace returns in `Trace.outputs` (`None` for one nothing wrote);
`Trace.output` renders the first. With outputs, several terminals may end the
run, joined at the end. The trace of a version 2 run carries `run_id` and
`graph_id` (`describe()["id"]`). `on_event` receives an `Event` for each
visit's `start` and `end` and each item's `item_start` and `item_end`, on the
run's threads; a listener that raises is warned about once and dropped.

`MAS(..., retry={"render": {"outage": 1}})` runs an agent or code node again for a
kind in `RETRYABLE`, on the same visit: one record, with `retries` and the cost
of every attempt. Two reads are reserved and written by no node: `error`, the
`{kind, message, node}` of the failure an `on error` edge routed to the reading
node, and `run`, `{cost_usd, elapsed_s, calls, records}` at read time.
`Agent(..., inputs={"slide": "page"})` fills a placeholder from a value of
another name, and records key their inputs by placeholder.
`MAS(..., prompt_names={"checker": "judge", "figure_checker": "judge"})` lets agents share one
prompt; `copy(prompts=...)` changes it for all of them, and an optimizer works on it as
one prompt, through the first agent that shares it, unless one of them is detached.

`describe()` gains `for each`, `placeholders` and `retry` on nodes, and
`outputs` and `groups` (one per chain) at the top, each only when set.

::: xmas.chains
    options:
      members:
        - ForEach
        - Chain
        - ChainShape
        - find_chains
        - collapse
        - ITEM_BUDGET_KEYS

## Version 2: media values

A value of type `image`, `audio`, `document`, `table` or `file`, or a list of
them, is a `Ref`: a location, a media type, a size and a sha256. State,
records and traces hold refs, never bytes; `Trace.to_dict()` writes each ref
as its `to_dict()`. `run_state` turns a `Path` input into `Ref.from_path` (a
reference to your file, whose checksum `read_bytes` checks on every read). A
code node that returns a `Path` writes a copy into the store; a `Ref` passes; bytes
are refused ("return a Path or a Ref, Ref.from_bytes stores bytes"). A model's
pictures and an MCP tool's image, audio and blob blocks go to the store too
(`MCPTools(..., writes={"paint": {"png": "image"}})`). The store is
`FileStore(XMAS_STORE)`, by default `~/.cache/xmas/store`, read on every
call and never a spec key (a relative path is taken from the working
directory); it writes `<root>/<sha[:2]>/<sha><ext>` once, through a
temporary file, with the mode `open()` would give it.

**Capabilities.** What a model takes and gives comes from the first of: the
system's `models` entry for its binding name (`MAS(..., models={...})` or a
spec's `models: {vision-model: {accepts: [image]}}`), the model object's
`accepts` and `produces` attributes, or text only. `MAS.capabilities(agent)`
answers it. A `models` entry holds `accepts` and `produces` only. Text and
json are always taken and given. `check_system` reports, among its other
problems:

- `agent 'judge' reads image 'png', but model 'text-model' accepts only text;
  bind a model that accepts image (models: {text-model: {accepts: [image]}})
  or convert it first with a code node`; for the default model the fix reads
  `give it accepts = ("text", "image")`.
- `agent 'writer' reads document 'house'; a model reads a document through a
  conversion code node, such as one using xmas.convert.document_text` (a table:
  `table_rows`). A `file` read reaches the prompt as the ref's JSON line.
- `agent 'painter' writes image 'picture', but model 'text-model' produces
  only text; bind a model that produces image (models: {text-model:
  {produces: [image]}})`, and `agent 'painter' writes image and has tools; an
  image answer takes none`. An agent writing audio, a document, a table or a
  file is refused: a model answers with text, json or image.
- `model 'vision-model' accepts image but cannot send content parts; use
  APILLM, or give it generate_parts_with_meta and full_response = True`, and
  the same for `produces image but cannot answer with images` and
  `generate_image_with_meta`. A subclass of `APILLM` that overrides a hook of
  promptolution's path has `full_response` false and is refused here.
- `models names 'x', which no agent binds`; `models entry 'x' has key 'colour';
  it takes model, accepts, produces and the parameters max_tokens, timeout_s,
  temperature, top_p, seed, reasoning_effort`; `models entry 'x' accepts
  'video'; a model may accept image, audio`.
- `code node 'draw' passes image 'png' to remote tool 'paint'; remote tools read no
  media`.
- `Agent 'reader' uses tool 'document_text', which reads document 'source'; a
  model passes no media to a tool, so run it as a code node.` A model would make up
  the ref, and the tool would read whatever file it names; this holds in
  version 1 too.

**Parts.** An agent that reads an image or audio renders its prompt with
`render_parts`: the text as `safe_format` gives it, with each ref in its
place. A list labels its items `[pages 0]`, `[pages 1]`, ... (0-based, as
`item`), a list of lists `[pages 1.0]`, and a `None` item reads `(none)`.
The record's `rendered_prompt` holds `marker` lines such as
`[image slide-1.png, image/png, 182 KB]` in place of the refs, and its `inputs` the refs' JSON lines; a json agent's format lines
join the last text part. `to_content` makes the request content (a data URI or
the URL of a remote image, `input_audio` for wav and mp3) and refuses a part
over `MAX_PART_BYTES`. A tool agent sends the parts in its first message. An
agent with `output_type="image"` calls `generate_image_with_meta`; an `image`
write needs exactly one picture and a `list of image` at least one, else the
answer is `bad_format`.

**Conversions.** `xmas.convert.document_text` (text, Markdown and HTML) and
`table_rows` (CSV and TSV) are callables a code node uses, so a model reads a
document or a table only through a visible node. Any other type is refused
with a message that asks for a step of your own.

::: xmas.media
    options:
      members:
        - Ref
        - Store
        - FileStore
        - store
        - Kind
        - Image
        - Audio
        - Document
        - Table
        - File
        - Capabilities
        - TEXT_ONLY
        - REF_KINDS
        - DOCUMENT_TYPES
        - TABLE_TYPES
        - ACCEPTABLE
        - PRODUCIBLE
        - MAX_PART_BYTES
        - kind_of
        - fits_kind

::: xmas.llms.parts
    options:
      members:
        - Part
        - render_parts
        - record_text
        - marker
        - has_media
        - to_content
        - content_text

::: xmas.convert
    options:
      members:
        - document_text
        - table_rows

## Version 2: people, Stop, first of

A **human action** asks a person: `HumanAction(name, form, reads, writes, question=...,
options=, labels=, pair=, edits=, timeout_s=, default=)` with one of the four
`FORMS`, passed as `MAS(..., human_actions=[...])`. A human makes the system version 2.
Its question and labels are templates over its reads, paths included. It compiles
to its asking node and a deferred `<name>.wait` node, so a question waits only
once every other branch has ended or paused; the edges out of a human leave from
its wait node and route on the answer. A chain or module that holds a human runs
its items or inner run on threads of their own, so one paused item never holds
its siblings.

**`MAS.run_state(values, on_event=None, run_id=None)`** returns a `paused`
trace (`Trace.pause`: a `Pause` with each open `Question`, its id
`<run_id>/<node>[/<item key or index>]/<visit>`) when questions are still open;
**`MAS.answer(run_id, answers)`** keeps answers (`{question id: answer}`) while
the run goes or waits, all or none ("The answers were not kept: ..." lists each
problem); **`MAS.resume(run_id, answers=None, on_event=None)`** goes on from
the run's checkpoints, in this process or a new one. It refuses an unknown run,
one that is not paused ("Run 'r' is stopped, not paused."), a system whose
`describe()` id differs ("resume with that system"), and a resume with no
question answered or past its deadline. The budget counts on from the pause
(`RunMeter.seed`); the wait is not wall time and is reported as `wait_s` on the
human's record and `waited_s` in health. A question past its deadline takes its
`default`, with `timed_out` true. An `Asker` (`MAS(..., asker=...)`) is asked
once per question and may answer at once; `withdraw(question, reason)` is
called for each open question of a run that is stopped, paused or running. `run`,
`run_batch`, `predict` and the optimizers never wait: an open question takes its
default, else the run fails with `Unanswered: human action 'x' got no answer in an
unattended run`, from a chain item or a box too. `ReplayAsker(traces)` answers from recorded human records.

The **run store** is a LangGraph checkpoint saver on one SQLite file, its
tables laid out as `langgraph-checkpoint-sqlite` lays them out, plus a row per
run (status, open questions, meter totals) and per question (payload, deadline,
answer). By default it is `${XDG_DATA_HOME:-~/.local/share}/xmas/runs.sqlite`
(`runs_path()`), made on first use; `XMAS_RUNS`, `[store] runs` in the
configuration file, a path or `MAS(..., store=RunStore(path))` put it elsewhere,
and `:memory:` keeps it in memory, which logs nothing. `RunStore.due()` lists
runs with a question past its deadline for a host's timer, and
`prune_runs(days)` and `delete(run_id)` clean up. Only the classes in
`XMAS_TYPES` are read back from a checkpoint.

The store is also the **run log**. Every `run`, `run_batch`, `run_state`,
`resume` and `predict`, version 1 included, writes through it as it goes
(`RunLog`, on the run's meter): its row in `xmas_runs` when it starts (`begin`:
run id, `describe()` id, `running`, start time) and when it ends or pauses
(status, totals, end time, query, output, error, the records' order); one row
in `xmas_records` the moment each record is kept (position, node, kind, item,
visit, model, cost, latency, tokens, finish reason, error, route, inputs and
outputs as JSON, the raw answer, the rendered prompt, and the record itself;
`boxed` marks a module's inner record); one in `xmas_events` per event; one in
`xmas_calls` per model call (call id, node, start time, model, served model,
tokens, cost, latency, finish reason, error kind, and the call's parameters
without any key, token, password or header). Rows are only added, one insert
each, so a run whose process dies keeps what it wrote; a write that fails warns
once and the run goes on. `RunStore.trace(run_id)` rebuilds the trace (its
health summarized again, with default settings), `RunStore.runs(status=None,
system_id=None, limit=None)` lists runs, and `events(run_id)` and
`calls(run_id)` read the rest. A run id given to `run_state` names one run in
the file. From the shell, `xmas runs [--json] [--status s] [--system id]
[--limit n]` lists runs, and `xmas trace <run id> [--json]` shows one run's
records as a table, or its whole trace.

**`MAS.stop(run_id, reason="stopped")`** stops a running run in this process
(`RUNNING` holds each run's meter by id): no call, code node or item starts after
it, waits for a slot or a gate end within `SLICE_S`, and a model call in flight
sees the stop on its `cancel_flag()` and is cut (`Stopped`, with the paid
`meta` and the partial `text`). The trace's status is `stopped`, one `stop`
event is emitted, and a visit cut after a call keeps a failed record. A paused
run is marked stopped instead; `False` means no such run is running or paused.
A run id given to `run_state` that is running ("is in use by a running run") or
kept in the store ("is taken") is refused, and so is a `/` in the id of a run
that can pause, since it names the threads of the run's items and modules. `APILLM(stream=True)` streams, so a
cut call keeps its partial answer; the default request is unchanged.

**`first of`** edges (`EdgeKind.FIRST`, `a -> b: first of` in a spec) from one
node form a race of agents, code nodes or modules that write the same values and go on
to one node or all end the run; a module holding a human does not race. Each
contestant runs under
`contestant(meter.race(key))`, the key being the run (each module visit and chain
item has its own), the source, the item and the visit;
the first to finish wins (`RunMeter.win`) and writes, the others are cut
(`RaceLost`) or their writes dropped, and each keeps a record with `lost` set
and its paid cost, its `end` event `lost`. A contestant that fails goes no
further; the race fails with the last failure when every contestant failed.

::: xmas.human
    options:
      members:
        - Question
        - Pause
        - Asker
        - ReplayAsker
        - answer_value
        - question_id
        - FORMS
        - ANSWER_FIELDS
        - COMMON_FIELDS
        - TIE
        - DEFAULT_LABELS

::: xmas.human.HumanAction
    options:
      heading_level: 3
      show_root_full_path: false
      members:
        - answer
        - renamed
        - written
        - __init__
        - ask
        - from_dict
        - problems
        - state_writes
        - templates
        - to_dict

::: xmas.store
    options:
      members:
        - RunRow
        - QuestionRow
        - open_store
        - XMAS_TYPES

::: xmas.store.Store
    options:
      heading_level: 3
      show_root_full_path: false
      members:
        - __init__
        - __repr__
        - answer
        - begin
        - calls
        - close
        - delete
        - delete_thread
        - due
        - events
        - get_next_version
        - get_tuple
        - list
        - prune_runs
        - put
        - put_writes
        - question
        - questions
        - run
        - runs
        - save_question
        - save_run
        - trace

::: xmas.store.RunLog
    options:
      heading_level: 3
      show_root_full_path: false
      members:
        - __init__
        - call
        - end
        - event
        - record

::: xmas.budget
    options:
      members:
        - CancelFlag
        - cancel_flag
        - contestant
        - RUNNING
        - SLICE_S

::: xmas.budget.RunMeter
    options:
      heading_level: 3
      show_root_full_path: false
      members:
        - elapsed_s
        - __init__
        - adopt
        - after_call
        - before_call
        - check_boundary
        - close
        - emit
        - fail
        - flag
        - gate
        - keep
        - metered
        - owns
        - race
        - records_of
        - seed
        - slot
        - start
        - stop
        - totals
        - win

## Bundles, verify, configuration, the command line

**`MAS.save(path, *, zipped=False)`** writes a system as a `.mas` bundle, a
folder or one zip file: `xmas.yaml` (the spec without its prompts),
`callables.yaml`, `prompts/<name>.md` (the text, after an optional header that
holds only the answer's `schema`), `modules/<box>.mas/`, and `xmas.lock` (the
sha256 of every other file and of each python callable's source), written last.
Into an existing folder it writes only the files whose parsed content changed
and never touches `examples/` or `optimize.yaml`. **`MAS.load(path, *, llm=None,
llms=None, tools=(), config=None, overlay=None, resolve=True, verify=None,
trust=False)`** reads one back: a callable in `tools` wins, a python callable is
found by its `import` path and bound to its `params`, a remote one reaches the
server its `server` names under `[tools.<name>]`, and every bound model name not
given gets `build_llms`, each module's `default_model` too. A bundle may name any
code, so without `trust` nothing is imported: a python callable must be a `Tool`
(made with `@tool`) in a module already imported, else the load refuses and
names `tools=[...]` and `trust=True`. With `resolve`, the names are checked
against the catalogue: a miss warns with a `ModelNameWarning`, lands in
`mas.unresolved`, and every run entry point, `resume` included, refuses until it
is fixed. A load refuses absolute paths, `..`, symlinks, a zip entry there
twice, a broken zip and bundles over `MAX_BYTES`; a lock whose bundle digest is
not that of its files is a level 0 difference.

**`MAS.verify(level=1, *, budget=None, strict=False, only=(), examples=None,
config=None)`** returns a `VerifyReport`: level 0 compares the bundle with its
lock, level 1 checks the system, renders every prompt, reads every examples file
and resolves every model name with no model call, level 2 runs the examples
under `BUDGET_USD` in all and `PER_EXAMPLE_USD` each unless a budget is given. A
report that is not ok warns once, or raises `VerifyFailed` when `strict`.
**`MAS.save_examples(trace, *, nodes=None, file="warmup", path=None)`** records
one node case per node from a trace into `examples/<file>.yaml`.

::: xmas.bundle
    options:
      members:
        - files_of
        - read
        - write
        - spec_of
        - dump_prompt
        - parse_prompt
        - read_optimize
        - Lock
        - lock_of
        - code_hashes
        - LockDiff
        - compare

::: xmas.examples
    options:
      members:
        - Check
        - Example
        - ExampleFile
        - parse_examples
        - run_checks
        - cases_from_trace
        - save_examples
        - CHECK_KINDS

::: xmas.verify
    options:
      members:
        - verify
        - ExampleResult
        - VerifyWarning
        - VerifyFailed
        - BUDGET_USD
        - PER_EXAMPLE_USD

::: xmas.verify.VerifyReport
    options:
      heading_level: 3
      show_root_full_path: false
      members:
        - ok
        - __str__
        - lines
        - to_dict
        - warn

::: xmas.config
    options:
      members:
        - Config
        - ToolServer
        - config_path
        - runs_path
        - Catalogue
        - read_catalogue
        - closest
        - misses
        - resolve_names
        - Resolution
        - ModelNameWarning
        - ModelEntry
        - PARAMS
        - build_llms
        - Unconfigured
        - tool_sources

::: xmas.cli
    options:
      members:
        - main

## Prompts

::: xmas.prompts
    options:
      members:
        - safe_format
        - encode_prompts
        - decode_prompts

## Tasks

::: xmas.tasks.base.MASTask

::: xmas.tasks.gsm8k.GSM8KTask

::: xmas.tasks.gsm8k.extract_number

## Optimization

::: xmas.optimization.optimizer.MASOptimizer

::: xmas.optimization.attribution.Attributor

::: xmas.optimization.attribution.UniformAttributor

::: xmas.optimization.attribution.ContrastiveAttributor

::: xmas.optimization.updaters.PromptUpdater

::: xmas.optimization.updaters.AgentPromptView

::: xmas.optimization.prefix_search.PrefixSearchOptimizer

### CANTANTE

Attribution-guided prompt optimization: every agent's prompt is improved at once,
by its own promptolution optimizer, against a single shared system evaluation
that an attributor splits back into per-agent credit.

::: xmas.optimization.cantante.CantanteOptimizer

::: xmas.optimization.broker
    options:
      members:
        - Board
        - ProxyTask
        - Request
        - rounds

### Topology search

The other half of the decomposition: the graph is the component, and the same
loop improves it. Operators work on the agent-level edge set, with the sentinel
wiring left to `build_graph`, so a candidate is legal exactly when a `MAS` can be
built from it. On a version 2 system they move only what `Fixed` leaves free,
and the system's own check decides which candidates are returned.

::: xmas.optimization.evolution.EvolutionaryTopologyOptimizer

::: xmas.optimization.graph_updater.GraphUpdater

::: xmas.optimization.topology
    options:
      members:
        - TopologyBudget
        - Fixed
        - mutate
        - crossover
        - repair
        - mas_edges
        - is_valid
        - within_budget
        - sources
        - terminals
        - cycles

### Prompt synthesis

Topology search moves information around; something has to write the sentence
that uses it.

::: xmas.optimization.synthesis.LLMPromptSynthesizer

::: xmas.optimization.synthesis.MechanicalSynthesizer

::: xmas.optimization.synthesis.Rewrite

## Serialization

::: xmas.spec
    options:
      members:
        - validate_spec
        - validate_spec_v2
