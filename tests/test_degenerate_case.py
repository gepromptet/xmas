"""The N=1 case: a one-agent MAS is a promptolution predictor, not an analogue.

This is the founding premise made executable. If these tests fail, xmas is a
lookalike of promptolution rather than a lift of it.
"""

from __future__ import annotations

import pandas as pd
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.prompt import Prompt

from xmas import MAS, Agent
from xmas.prompts import decode_prompts, encode_prompts

from .conftest import MockLLM


def single_agent_mas(llm: MockLLM, prompt: str = "Answer the question.") -> MAS:
    """A MAS with exactly one agent, which is the degenerate case."""
    return MAS([Agent("solver", prompt=prompt, output="prediction")], llm=llm)


def test_a_mas_is_a_promptolution_predictor(mock_llm: MockLLM) -> None:
    assert isinstance(single_agent_mas(mock_llm), BasePredictor)


def test_one_agent_prompt_set_is_the_bare_string() -> None:
    """No wrapper, no header: byte-identical to a promptolution prompt."""
    encoded = encode_prompts({"solver": "Answer the question."})
    assert encoded == "Answer the question."
    assert decode_prompts(encoded, ["solver"]) == {"solver": "Answer the question."}


def test_a_one_node_mas_runs_inside_a_plain_promptolution_task() -> None:
    """The load-bearing test: promptolution drives xmas with no adapter."""
    labels = ["positive", "negative", "positive"]
    df = pd.DataFrame({"x": ["great", "awful", "lovely"], "y": labels})

    # The model answers with the label the task expects, so a correct wiring
    # scores 1.0 and a broken one cannot accidentally look right.
    lookup = {"great": "positive", "awful": "negative", "lovely": "positive"}
    llm = MockLLM(lambda prompt: next(v for k, v in lookup.items() if k in prompt))

    mas = single_agent_mas(llm, prompt="Classify: {query}")
    task = ClassificationTask(df, x_column="x", y_column="y", task_description="sentiment")

    result = task.evaluate(Prompt("Classify: {query}"), predictor=mas)

    assert result.agg_scores.shape == (1,)
    assert result.agg_scores[0] == 1.0


def test_promptolution_evaluates_several_prompts_against_the_same_mas() -> None:
    """What an optimizer does: score a population of prompts through one predictor."""
    df = pd.DataFrame({"x": ["a", "b"], "y": ["yes", "yes"]})
    llm = MockLLM(lambda prompt: "yes" if "GOOD" in prompt else "no")
    mas = single_agent_mas(llm)
    task = ClassificationTask(df, x_column="x", y_column="y")

    result = task.evaluate([Prompt("GOOD instruction"), Prompt("BAD instruction")], predictor=mas)

    assert result.agg_scores.tolist() == [1.0, 0.0]


def test_predict_honours_the_flattened_prompt_and_input_pairing() -> None:
    """Tasks flatten (prompt, x) pairs, so predict must respect the pairing."""
    llm = MockLLM(lambda prompt: prompt)
    mas = single_agent_mas(llm)

    preds, seqs = mas.predict(prompts=["P1", "P2"], xs=["x1", "x2"])

    assert preds == ["P1", "P2"]
    assert len(seqs) == 2


def test_predict_tolerates_an_empty_batch(mock_llm: MockLLM) -> None:
    """A warm task cache calls predict with nothing to do."""
    assert single_agent_mas(mock_llm).predict(prompts=[], xs=[]) == ([], [])
