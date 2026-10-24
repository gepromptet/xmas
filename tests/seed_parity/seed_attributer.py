# Vendored verbatim from tritthart/mas-optimization main (2db997d),
# mas_lib/attribution/{base,absolute}.py, itself adapted from
# https://github.com/finitearth/cantante. See README.md for the three
# substitutions and the names without a leading underscore. Do not edit to
# make a test pass.
# ruff: noqa
# mypy: ignore-errors

import logging

logger = logging.getLogger(__name__)


def constrain_prompt_length(prompt, max_tokens):
    """Stub: the seed truncates by tiktoken token count; the fixture never reaches the budget."""
    return prompt


class AgentPromptBatch:
    """The one method the attributer calls on the seed's batch object."""

    def __init__(self, agent_names):
        self.agent_names = list(agent_names)

    def get_agent_names(self):
        return list(self.agent_names)


ATTRIBUTION_PROMPT = """# Task
You are an attribution agent for a multi-agent system.

You will receive multiple executions of the SAME query.
Each execution uses different agent parametrizations but follows the same workflow.

Your goal is to estimate how each agent's behavior contributed to the outcome of its execution by comparing differences across executions.

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
- Identical predictions or system scores do not imply identical contributions; base attribution on differences in reasoning and role adherence.
- The "query" may include example tests (e.g. `assert f(...) == ...`). These define the EXACT required behaviour and output format the system is graded on. Judge correctness strictly against those tests, NOT against your own notion of what is "mathematically" or "conceptually" right. An output that looks reasonable but violates the example tests (wrong return type, wrong format, wrong value) is INCORRECT and must not receive positive credit; the agent responsible for that deviation should receive negative credit.

## Input
You will receive JSON with:
- "query": same for all executions; may embed the example tests that define the grading contract
- "executions": list of execution results, each containing:
  - "system_score": this is the reward signal for attribution, computed by the evaluator based on the execution's prediction. The goal is to maximize this quantity.
  - "agent_outputs": mapping from agent name to that agent's output. The output with the key "prediction" is the final prediction, on which the system is scored on.

## Output
First, briefly compare the executions and identify the key behavioral differences that lead to differences in the system score.

Then return JSON inside <attribution>...</attribution>.

## Critical Formatting Rules
- The content inside <attribution>...</attribution> must be valid JSON.
- Do not use markdown code fences inside the <attribution> block.
- The top-level JSON object inside <attribution> must have exactly one key: "executions".
- "executions" must be a list with exactly {n_parametrizations} items.
- Preserve the same execution order as given in the input.
- Each item in "executions" must be an object with exactly one key: "agent_credits".
- "agent_credits" must be an object containing exactly these agent names and no others:
  {agent_names}
- Every credit must be a float between -1.0 and 1.0.
- Do not omit any execution.
- Do not put explanations, summaries, or any other text inside the JSON.
- Any reasoning or explanation must be placed outside the <attribution> block.

# Few-Shot Example

## Input
{
  "query": "A number doubled and then increased by 3 equals 11. What is the number?",
  "executions": [
    {
      "system_score": 1.0,
      "agent_outputs": {
        "planner": {
            "plan": "Let x be the number. Then 2x + 3 = 11. Subtract 3 to get 2x = 8. Divide by 2 to get x = 4."
        },
        "formatter": {
            "formatted_answer": "4"
        },
        "critic": {
            "correct: "Yes",
            "prediction": "4"
        }
      }
    },
    {
      "system_score": 0.0,
      "agent_outputs": {
        "planner": {
            "plan": "A possible number is 5 because doubling gives 10 and that is close to 11."
        },
        "formatter": {
            "formatted_answer": "5"
        },
        "critic": {
            "correct: "True",
            "prediction": "5"
        }
      }
    },
    {
      "system_score": 0.0,
      "agent_outputs": {
        "planner": {
            "plan": "Let x be the number. Then 2x + 3 = 11. Subtract 3 to get 2x = 8. Divide by 2 to get x = 4."
        },
        "formatter": {
            "formatted_answer": "x = 4"
        },
        "critic": {
            "correct: "True",
            "prediction": "x = 4"
        }
      }
    }
  ]
}

## Output

The first execution is fully correct. The planner sets up and solves the equation properly, the formatter states the correct answer, and the critic confirms consistency between reasoning and output. An attribution of 1.0 for all agents is appropriate as all agents performed their roles without any error.

The second execution fails primarily due to the planner. The planner does not formulate the equation and instead relies on an unsupported guess, which leads to an incorrect solution. The formatter correctly follows its role by presenting the answer clearly and in the required format, but it propagates the incorrect result. The critic fails to identify that the reasoning is invalid and does not challenge the incorrect answer. Therefore, the planner deserves an attribution of -1.0, as it is the source of the error, while the critic shares some blame for not catching the error, giving an attribution of -0.8. The formatter, while it does propagate the error, is not the cause and still fulfills its role in terms of format and clarity, so an attribution of 1.0 is reasonable.

The third execution shows a localized failure due to formatting. The planner correctly solves the equation, and the critic confirms that the reasoning is valid. However, the formatter outputs the answer in an incorrect format, violating the output requirements. A system score of 0.4 is reasonable here: the underlying reasoning and value are correct, but the final output is not usable due to formatting errors.

<attribution>
{
  "executions": [
    {
      "agent_credits": {
        "planner": 1.0,
        "formatter": 1.0,
        "critic": 1.0
      }
    },
    {
      "agent_credits": {
        "planner": -1.0,
        "formatter": 1.0,
        "critic": -0.8
      }
    },
    {
      "agent_credits": {
        "planner": 1.0,
        "formatter": -1.0,
        "critic": 0.9
      }
    }
  ]
}
</attribution>

# Attribution

Now perform attribution for the following context.

## Query and Executions
{query_and_prompt_sets}

Compare all {n_parametrizations} executions and assign scores for agents in {agent_names}."""


# Adds measured environment effects to the default attribution context.


"""Base attributer interface."""

import re
from abc import ABC, abstractmethod


def build_agent_var_mapping(predictor):
    """Map agent names to their variables and task descriptions."""
    mapping = {}
    for agent_name, agent in predictor.agents.items():
        mapping[agent_name] = {
            "input_vars": agent.input_vars,
            "output_vars": agent.output_vars,
            "task_description": agent.task_description,
        }
    return mapping


class BaseAttributer(ABC):
    """Base interface for agent credit assignment."""

    def __init__(self, llm=None, agent_var_mapping=None, *args, **kwargs):
        self.llm = llm
        self.agent_var_mapping = agent_var_mapping or {}

    def clean_state(self, state, only_include=None):
        """Drop non-attribution fields from a single state dict (returns a copy)."""
        if state is None:
            return {}
        ignore = {"few_shots", "agent_prompt_set"}
        state = {k: v for k, v in state.copy().items() if k not in ignore}
        if only_include is not None:
            state = {k: v for k, v in state.items() if k in only_include}
        return state

    def state_to_agent_outputs(self, state):
        """Convert a state dict into {agent_name: {key: value}}."""
        state = self.clean_state(state)
        by_agent = {agent_name: {} for agent_name in self.agent_var_mapping}
        for agent_name, var_info in self.agent_var_mapping.items():
            for output_key in var_info["output_vars"]:
                if output_key in state:
                    by_agent[agent_name][output_key] = state[output_key]
        return by_agent

    def extract_tag(self, text, tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", str(text), flags=re.DOTALL)
        return m.group(1).strip() if m else ""

    # Line-structured fields must remain intact.
    NO_TRUNCATE_KEYS = frozenset()

    def truncate_field_value(self, value):
        if self.max_tokens_per_field is None:
            return value

        if isinstance(value, str):
            return constrain_prompt_length(value, self.max_tokens_per_field)

        if isinstance(value, dict):
            return {k: (v if k in self.NO_TRUNCATE_KEYS else self.truncate_field_value(v)) for k, v in value.items()}

        if isinstance(value, list):
            return [self.truncate_field_value(v) for v in value]

        return value

    @abstractmethod
    def attribute(
        self,
        agent_prompt_batch,
        system_scores,
        intermediate_results,
    ):
        """Return unaggregated per-agent scores aligned with ``system_scores``."""
        raise NotImplementedError("Subclasses must implement the attribute method.")


import json

import numpy as np


class AbsoluteAttributer(BaseAttributer):
    """Assign unconstrained per-query agent credit with an LLM."""

    def __init__(
        self,
        llm=None,
        agent_var_mapping=None,
        prompt_template=None,
        group_size=4,
        max_tokens_per_field=1024,
        seed=42,
    ):
        super().__init__(llm=llm, agent_var_mapping=agent_var_mapping)
        self.prompt_template = prompt_template or ATTRIBUTION_PROMPT
        self.group_size = group_size
        self.max_tokens_per_field = max_tokens_per_field
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def build_all_prompts(self, agent_names, system_scores, intermediate_results):
        """Build attribution prompts in groups of ``group_size`` runs."""
        n_queries = len(system_scores[0]) if len(system_scores) > 0 else 0
        n_runs = len(system_scores)

        prompts = []
        prompt_metadata = []

        for qi in range(n_queries):
            query = self.clean_state(intermediate_results[0][qi])["query"]

            for group_start in range(0, n_runs, self.group_size):
                group_end = min(group_start + self.group_size, n_runs)
                run_indices = list(range(group_start, group_end))

                shuffled_indices = run_indices.copy()
                self.rng.shuffle(shuffled_indices)

                shuffle_mapping = {i: shuffled_indices[i] for i in range(len(shuffled_indices))}

                rollouts = []
                for run_idx in shuffled_indices:
                    state = self.clean_state(intermediate_results[run_idx][qi])
                    rollouts.append(self.build_rollout(float(system_scores[run_idx][qi]), state))

                query_and_sets = {
                    "query": query,
                    "executions": rollouts,
                }

                query_and_sets = self.truncate_field_value(query_and_sets)

                prompt = self.safe_format_prompt(
                    self.prompt_template,
                    query_and_prompt_sets=json.dumps(query_and_sets, ensure_ascii=False, indent=2),
                    n_parametrizations=str(len(rollouts)),
                    agent_names=str(agent_names),
                )
                prompts.append(prompt)
                prompt_metadata.append((qi, run_indices, shuffle_mapping))

        prompts = [
            constrain_prompt_length(prompt, self.llm.max_tokens - self.max_tokens_per_field) for prompt in prompts
        ]

        return prompts, prompt_metadata

    def build_rollout(self, system_score, state):
        """Build one execution entry for the attribution prompt."""
        return {
            "system_score": system_score,
            "agent_outputs": self.state_to_agent_outputs(state),
        }

    def safe_format_prompt(self, prompt, **kwargs):
        """Format prompt template robustly via granular placeholder replacement."""
        for key, value in kwargs.items():
            placeholder = "{" + str(key) + "}"
            try:
                prompt = prompt.replace(placeholder, str(value))
            except Exception as e:
                logger.warning(f"Failed to replace placeholder '{key}': {e}")

        return prompt

    def parse_attributions_response(self, response, agent_names, group_size):
        """Parse an attribution response into a run-by-agent matrix."""
        attribution_matrix = np.zeros((group_size, len(agent_names)))

        payload = self.extract_tag(response, "attribution")
        try:
            attributions_by_run = json.loads(payload)
            attributions_by_run = attributions_by_run["executions"]
            for group_idx in range(group_size):
                attributions = attributions_by_run[group_idx]
                attributions = attributions["agent_credits"]
                attributions = np.array([float(attributions[a]) for a in agent_names])
                attribution_matrix[group_idx] = np.clip(attributions, -1, 1)

        except Exception as e:
            logger.warning(f"Failed to parse attribution weights from response: {response}")
            logger.warning(f"Error: {e}. Falling back to zeros.")

        return attribution_matrix

    def attribute(self, agent_prompt_batch, system_scores, intermediate_results):
        """Attribute scores for every run and query in the batch."""
        if self.llm is None:
            raise ValueError("llm must be provided for AbsoluteAttributer")
        agent_names = agent_prompt_batch.get_agent_names()

        n_runs = len(system_scores)
        n_queries = len(system_scores[0]) if len(system_scores) > 0 else 0
        prompts, prompt_metadata = self.build_all_prompts(agent_names, system_scores, intermediate_results)
        try:
            responses = self.llm.get_response(prompts)
        except Exception as e:
            logger.error(f"LLM call for attribution failed: {e}")
            responses = [""] * len(prompts)
        attributed = {a: [[None] * n_queries for _ in range(n_runs)] for a in agent_names}
        for response, (qi, run_indices, shuffle_mapping) in zip(responses, prompt_metadata):
            group_size = len(run_indices)
            attribution_matrix = self.parse_attributions_response(response, agent_names, group_size)

            for shuffled_idx in range(group_size):
                run_idx = shuffle_mapping[shuffled_idx]
                for ai, agent_name in enumerate(agent_names):
                    credit = attribution_matrix[shuffled_idx, ai]
                    attributed[agent_name][run_idx][qi] = credit

        return attributed
