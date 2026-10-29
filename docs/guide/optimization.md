# 🧠 Optimization built in

CANTANTE improves every agent's prompt at once, splitting each score into credit per agent.
Here it tunes the [quickstart](https://github.com/gepromptet/xmas#readme)'s two agents,
`team` from [Bundles](bundles.md).

```python
from promptolution.optimizers import OPRO
from xmas import GSM8KTask
from xmas.optimization import CantanteOptimizer, ContrastiveAttributor

task = GSM8KTask.from_huggingface(split="train", n=50)  # needs: uv run --with datasets
optimizer = CantanteOptimizer(team, task, ContrastiveAttributor(llm), OPRO, optimizer_kwargs={"meta_llm": llm})
best = optimizer.optimize(n_steps=5)
best.to_yaml("team-optimized.yaml")
```

`boss.detached = True` (or `Agent(..., detached=True)`, on any kind of node) keeps a node as it is.
No optimizer changes its prompt, model, reads, writes or edges. A detached node makes the system version 2, so keep its
`to_dict()` instead of a YAML file. `EvolutionaryTopologyOptimizer` searches the wiring.
It keeps code nodes, modules, human actions, detached nodes, conditional edges and every `for each`
chain as they are, and drops any candidate the check refuses. Code nodes are never part of the
prompts an optimizer sees, and agents that share a prompt are optimized as one prompt.

## 🔗 xmas and promptolution

xmas builds on [promptolution](https://github.com/gepromptet/promptolution): a system is a
promptolution predictor and a task is a promptolution task, so promptolution's optimizers
run on any agent unchanged.

A one-agent system encodes to its bare prompt, with or without code nodes. So promptolution's
optimizers send the same requests to it as to a single prompt.
