# xmas

xmas is the multi-agent sibling of
[promptolution](https://github.com/gepromptet/promptolution). It keeps the same five
nouns, Task, Predictor, LLM, Optimizer and ExperimentConfig, and lifts them one level:
the multi-agent system itself is the Predictor, and its per-agent prompts, its topology
and its per-node model bindings are all optimizable components. A single-agent system is
the exact degenerate case, so promptolution's optimizers plug in unchanged.

!!! note "Beta"

    Inference, the optimization loop, CANTANTE, topology search, code nodes, JSON
    answers, conditions, loops, per-item chains, media values, budgets, health,
    people in the graph, Stop, `first of` races, the `.mas` bundle, `verify()`,
    the run log and the `xmas` command are in place. Learning from human picks is
    still to come.

## Guide

The guide grows one small system, page by page, from the quickstart in the
[README](https://github.com/gepromptet/xmas#readme).

- [Steps and values](guide/steps-and-values.md): code between agents, and the check before any call.
- [Answers with a shape](guide/answer-forms.md): JSON answers with fixed fields.
- [Conditions and loops](guide/conditions-and-loops.md): `when`, `else`, `while`, `switch` and `on error` edges.
- [Reading a run](guide/reading-a-run.md): records, health, budgets and the run log.
- [Drawing a system](guide/drawing-a-system.md): `describe()`, the system as plain data.
- [A system inside a system](guide/boxes.md): a module runs a whole system as one node.
- [Once per item](guide/chains.md): `for each` chains.
- [Several values in and out](guide/run-state.md): `run_state`, events and retries.
- [Pictures and files](guide/media.md): media values, passed by reference.
- [People, Stop and races](guide/people.md): people in the graph, Stop and `first of`.
- [Bundles](guide/bundles.md): the `.mas` bundle, `verify()`, the `xmas` command and configuration.
- [Optimization](guide/optimization.md): CANTANTE, detached nodes and topology search.
- [Where this is going](guide/roadmap.md): what is planned and not built.

## More

- [Design principles](design-principles.md), the six rules the design answers to.
- [API reference](api.md), generated from docstrings.
- [Contributing](https://github.com/gepromptet/xmas/blob/main/CONTRIBUTING.md).
