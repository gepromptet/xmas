# API reference

Generated from docstrings. Everything on this page is public surface; anything
not on it is an implementation detail and may change without notice.

## Defining a system

::: xmas.agent.Agent

::: xmas.mas.MAS

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

## Models

Each backend subclasses its promptolution counterpart and adds `score()`, the
teacher-forced sibling of generation.

::: xmas.llms.api_llm.APILLM

::: xmas.llms.api_llm.ScoringUnsupportedError

::: xmas.llms.scoring.ScoredSequence

::: xmas.llms.masking
    options:
      members:
        - assistant_span
        - mask_logprobs

## Prompts

::: xmas.prompts
    options:
      members:
        - safe_format
        - extract_placeholders
        - encode_prompts
        - decode_prompts

## Tasks

::: xmas.tasks.base.MASTask

::: xmas.tasks.gsm8k.GSM8KTask

::: xmas.tasks.gsm8k.extract_number

## Optimization

::: xmas.optimization.base.MASOptimizer

::: xmas.optimization.attribution.Attributor

::: xmas.optimization.attribution.UniformAttributor

::: xmas.optimization.attribution.ContrastiveAttributor

::: xmas.optimization.attribution.UNKNOWN

::: xmas.optimization.updaters.Updater

::: xmas.optimization.updaters.TopologyUpdater

::: xmas.optimization.updaters.PromptUpdater

::: xmas.optimization.updaters.AgentPromptView

::: xmas.optimization.reference.PrefixSearchOptimizer

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
built from it.

::: xmas.optimization.evolution.EvolutionaryTopologyOptimizer

::: xmas.optimization.graph_updater.GraphUpdater

::: xmas.optimization.topology
    options:
      members:
        - TopologyBudget
        - mutate
        - crossover
        - repair
        - mas_edges
        - is_valid
        - within_budget
        - agent_edges
        - sources
        - terminals
        - cycles

### Prompt synthesis

Topology search moves information around; something has to write the sentence
that uses it.

::: xmas.optimization.synthesis.PromptSynthesizer

::: xmas.optimization.synthesis.LLMPromptSynthesizer

::: xmas.optimization.synthesis.MechanicalSynthesizer

::: xmas.optimization.synthesis.Rewrite

## Serialization

::: xmas.spec
    options:
      members:
        - MASSpec
        - AgentSpec
        - validate_spec
        - SPEC_VERSION
