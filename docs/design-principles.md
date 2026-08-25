# xmas design principles

Founding premise: single-prompt optimization is the N=1 case of multi-agent-system optimization. xmas is therefore promptolution's abstractions lifted one level, and the degenerate case must collapse back exactly.

**P1. Same five nouns, one lift.** Task, Predictor, LLM, Optimizer, and ExperimentConfig keep their promptolution meanings. The lift: the MAS is the Predictor, a structured predictor whose parameters (per-agent prompts, topology, per-node model bindings) are exposed as optimizable components. A one-node graph is a promptolution predictor.

**P2. Optimizers act through the Task, never around it.** The contract is a narrow evaluate; any optimizer runs on any task. Evaluation strategy (generation-based vs. likelihood probing) is a Task-level concern, invisible to optimizers: a probing Task can wrap any other Task and every optimizer runs unchanged.

**P3. One LLM interface owns all inference concerns.** Backends (API, vLLM, local), caching, concurrency, and token metering live in the LLM layer and nowhere else. xmas adds score() as generate()'s sibling (teacher-forced logprob scoring of a fixed continuation, with assistant-token masking as a standalone utility) and per-node model binding. score() is the framework's one probing primitive: ranking a fixed set of options by their first token is a different measurement and is deliberately not part of the research path, so no interface for it exists here.

**P4. New methods are new components, not new frameworks.** The optimization loop decomposes into Attributors (rollout plus reward to per-component credit) and per-component-type Updaters (a promptolution optimizer is a prompt Updater for one node; topology optimization is a graph Updater), orchestrated by a MAS optimizer. New research ships as a new Attributor or a new Updater.

**P5. Library-first, with a thin one-call experiment layer.** A clean Python API for full control, plus an ExperimentConfig and run-experiment mirror for reproducible large-scale runs. Output is framework-agnostic data: a plain serializable MAS spec (prompts, graph, model bindings), not code.

**P6. Lightweight through extras.** Minimal core dependencies; heavy backends behind optional extras, following promptolution's [api]/[vllm]/[transformers] pattern. Caching and usage tracking are core because they are identity features, not weight.
