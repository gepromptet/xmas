# Adapted from CANTANTE by Tom Zehle: https://github.com/finitearth/cantante
"""Meta-prompts the optimization components send to a meta-model.

These are method content, not configuration. The attribution prompt is the
CANTANTE contrastive prompt with its benchmark-specific paragraph removed (it
instructed the model to grade against MBPP-style ``assert`` tests embedded in the
query) and with the terminal output variable named rather than hardcoded to
``prediction``.
"""

ATTRIBUTION_PROMPT = """# Task
You are an attribution agent for a multi-agent system.

You will receive multiple executions of the SAME query.
Each execution uses different agent parametrizations but follows the same workflow.

Your goal is to estimate how each agent's behavior contributed to the outcome of its execution by comparing \
differences across executions.

Base your attribution on:
- the system score differences across executions,
- the system prediction
- how agent outputs differ across executions,
- whether these differences improve or degrade downstream reasoning,
- whether an agent introduces useful reasoning or merely follows,
- whether an agent propagates, corrects, or ignores errors.

## Attribution Scale
Assign a float score in [-1.0, 1.0] to each agent for each execution:
- positive = helpful contribution
- negative = harmful contribution
- 0 = neutral or unclear impact

## Critical Reasoning Guidelines
- Identical predictions or system scores do not imply identical contributions; base attribution on differences \
in reasoning and role adherence.

## Input
You will receive JSON with:
- "query": same for all executions
- "executions": list of execution results, each containing:
  - "system_score": this is the reward signal for attribution, computed by the evaluator based on the \
execution's prediction. The goal is to maximize this quantity.
  - "agent_outputs": mapping from agent name to that agent's output. The output written by the terminal agent, \
under the key "{output_variable}", is the final prediction, on which the system is scored.

## Output
First, briefly compare the executions and identify the key behavioral differences that lead to differences in \
the system score.

Then return JSON inside <attribution>...</attribution>.

## Critical Formatting Rules
- The content inside <attribution>...</attribution> must be valid JSON.
- Do not use markdown code fences inside the <attribution> block.
- The top-level JSON object inside <attribution> must have exactly one key: "executions".
- "executions" must be a list with exactly {n_executions} items.
- Preserve the same execution order as given in the input.
- Each item in "executions" must be an object with exactly one key: "agent_credits".
- "agent_credits" must be an object containing exactly these agent names and no others:
  {agent_names}
- Every credit must be a float between -1.0 and 1.0.
- Do not omit any execution.
- Do not put explanations, summaries, or any other text inside the JSON.
- Any reasoning or explanation must be placed outside the <attribution> block.

# Input

{executions}
"""
"""Contrastive credit assignment over several executions of one query."""

REWRITE_PROMPT = """# Task
You are rewriting the prompt of one agent in a multi-agent system, because the system's topology changed and \
the agent now receives different information than it did before.

Keep the agent's job the same. Change only what is needed for it to use everything it now receives.

## The agent
Name: {agent}
It writes: {output}
It now receives these variables: {inputs}

## Its current prompt
{prompt}

## Rules
- Reference every variable it receives as a placeholder in curly braces, exactly as named above.
- Do not invent placeholders for variables that are not listed.
- Keep the instruction concise and in the same voice as the current prompt.
- Return only the rewritten prompt inside <prompt>...</prompt>, with no commentary.
"""
"""Rewrite one agent's prompt after its inputs changed, used by topology search."""
