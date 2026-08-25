# API reference

Generated from docstrings. Everything on this page is public surface; anything
not on it is an implementation detail and may change without notice.

## Defining a system

::: xmas.agent.Agent

::: xmas.mas.MAS

## Topology

The graph is a plain `networkx.DiGraph` over agent names plus two sentinels, so a
topology optimizer is an ordinary graph-rewriting component.

::: xmas.graph
    options:
      members:
        - START
        - END
        - build_graph
        - validate_graph
        - agent_names
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

::: xmas.optimization.updaters.Updater

::: xmas.optimization.updaters.TopologyUpdater

::: xmas.optimization.updaters.PromptUpdater

::: xmas.optimization.updaters.AgentPromptView

::: xmas.optimization.reference.PrefixSearchOptimizer

## Serialization

::: xmas.spec
    options:
      members:
        - MASSpec
        - AgentSpec
        - validate_spec
        - SPEC_VERSION
