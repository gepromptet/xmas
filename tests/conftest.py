"""Shared test doubles: scripted models, a recording tool source, a patched APILLM, and the worked examples.

The mock model is the reason CI can run the whole framework end to end without a
key: it satisfies promptolution's LLM contract exactly, so nothing under test has
to know it is not talking to a real endpoint.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
import yaml
from openai.types.chat import ChatCompletion
from promptolution.llms.base_llm import BaseLLM

from xmas import MAS, Agent, TokenUsage
from xmas.edges import Edge, EdgeKind
from xmas.llms.api_llm import APILLM
from xmas.llms.meta import CallMeta
from xmas.media import Image, Ref
from xmas.nodes import Code
from xmas.tools import Tool, ToolCallRequest, ToolSource, tool

Turn = tuple[str, list[ToolCallRequest]]
"""One tool turn: the model's text and the calls it requests."""

META = CallMeta(latency_s=0.5, tokens=TokenUsage(10, 2), cost_usd=0.01)
"""What a ``MockLLM(meta=True)`` reports for every call."""


class MockLLM(BaseLLM):
    """A deterministic model that records every prompt it was given.

    It answers each prompt with ``responder`` (by default it echoes the prompt
    back, which makes substitution bugs visible in output), or with the next of
    ``answers``, raising an exception found there. Built with ``turns`` it also takes tool turns from that script,
    and ``default`` once the script runs out; built without, it has no
    ``get_tool_response``, as a plain promptolution model has none. With
    ``meta=True`` it reports :data:`META` for every call, so a meter sums cost.
    """

    get_tool_response: Callable[[list[dict[str, Any]], Sequence[Tool]], Turn]

    def __init__(
        self,
        responder: Callable[[str], str] | None = None,
        *,
        answers: list[str | Exception] | None = None,
        turns: list[Turn] | None = None,
        default: Turn = ("done", []),
        meta: bool = False,
    ) -> None:
        """Script the answers and, optionally, the tool turns and the call metadata."""
        super().__init__()
        self.answers = list(answers or [])
        self.responder = responder or (self.next_answer if answers is not None else lambda prompt: f"echo:{prompt}")
        self.calls: list[str] = []
        self.turns = list(turns or [])
        self.default = default
        self.seen: list[tuple[list[dict[str, Any]], list[str]]] = []
        self.full_response = meta
        if turns is not None:
            self.get_tool_response = self.tool_turn

    def next_answer(self, prompt: str) -> str:
        """Return the next scripted answer, raising it when it is an exception."""
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        """Return one scripted response per prompt."""
        self.calls.extend(prompts)
        return [self.responder(prompt) for prompt in prompts]

    def tool_turn(self, messages: list[dict[str, Any]], tools: Sequence[Tool]) -> Turn:
        """Record a snapshot of the history and the tools offered, and answer from the script."""
        self.seen.append((json.loads(json.dumps(messages)), [t.name for t in tools]))
        return self.turns.pop(0) if self.turns else self.default

    def generate_with_meta(self, prompt: str, system_prompt: str = "") -> tuple[str, CallMeta]:
        """Record the prompt and answer it, with metadata."""
        self.calls.append(prompt)
        return self.responder(prompt), META

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Answer from the tool script, with metadata."""
        text, requests = self.get_tool_response(messages, tools)
        return text, requests, META


@pytest.fixture(autouse=True)
def run_log_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Every test logs its runs under a data folder of its own, never into the home folder; yield that folder.

    The patch is the fixture's own, so a test's ``monkeypatch.undo()`` keeps it.
    """
    home = tmp_path_factory.mktemp("data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_DATA_HOME", str(home))
        patch.setenv("XDG_CONFIG_HOME", str(home / "no-config"))
        patch.delenv("XMAS_RUNS", raising=False)
        patch.delenv("XMAS_CONFIG", raising=False)
        yield home


@pytest.fixture
def mock_llm() -> MockLLM:
    """A model that echoes its prompt."""
    return MockLLM()


def req(name: str, arguments: str = "{}", call_id: str | None = None) -> ToolCallRequest:
    """A requested tool call; its id defaults to ``call_<name>``."""
    return ToolCallRequest(call_id or f"call_{name}", name, arguments)


class FakeToolSource(ToolSource):
    """A ToolSource with scripted answers; records every call."""

    def __init__(
        self,
        names: Sequence[str] = ("run_python",),
        answers: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Offer ``names``; ``answers`` maps a name to its result, or to an exception to raise.

        ``properties`` is the JSON Schema ``properties`` every tool declares.
        """
        schema = {"type": "object", "properties": dict(properties or {})}
        self.specs = [Tool(name, f"{name} tool", schema) for name in names]
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listed = 0

    def tools(self) -> list[Tool]:
        """Return the tools, counting how often they were listed."""
        self.listed += 1
        return list(self.specs)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Record the call and answer it."""
        self.calls.append((name, arguments))
        answer = self.answers.get(name, f"{name} ok")
        if isinstance(answer, Exception):
            raise answer
        return str(answer)


@tool
def add(a: int, b: int = 2) -> int:
    """Add two integers.

    Longer prose that is not part of the description.

    Args:
        a: The first summand.
        b: The second summand,
            continued on a second line.

    Returns:
        The sum.
    """
    return a + b


# ------------------------------------------------------------ APILLM doubles

BASE_URL = "https://gateway.example/v1"
URL = f"{BASE_URL}/chat/completions"


def status_error(
    cls: type[openai.APIStatusError], status: int, message: str = "", body: object | None = None
) -> openai.APIStatusError:
    """An openai error as the client raises it for an HTTP error status."""
    response = httpx2.Response(status, request=httpx2.Request("POST", URL))
    return cls(message or f"Error code: {status}", response=response, body=body)


def answer(
    content: str | None,
    *,
    finish: str | None = None,
    calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    model: str = "m",
    **extra: Any,
) -> ChatCompletion:
    """A chat completion as a provider would send it; ``extra`` lands at the top level.

    ``finish`` defaults to ``tool_calls`` when there are calls, else ``stop``.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    body: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish or ("tool_calls" if calls else "stop"), "message": message}],
        **extra,
    }
    if usage is not None:
        body["usage"] = usage
    return ChatCompletion.model_validate(body)


def recording_llm(
    monkeypatch: pytest.MonkeyPatch, answers: list[Any], cls: type[APILLM] = APILLM, **kwargs: Any
) -> tuple[APILLM, list[dict[str, Any]]]:
    """An APILLM whose client records each request, as sent, and replays ``answers``; an exception is raised."""
    llm = cls(model="m", base_url=BASE_URL, api_key="k", retry_base_delay_s=0, **kwargs)
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> ChatCompletion:
        sent.append(copy.deepcopy(request))  # the tool loop appends to the same messages list later
        reply = answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        assert isinstance(reply, ChatCompletion)
        return reply

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    return llm, sent


# ------------------------------------------------------- the worked example


@tool(writes={"counts": "json"})
def count_words(draft: str) -> dict[str, int]:
    """Count the words of a draft."""
    return {"words": len(draft.split())}


@tool(writes={"published": "text"})
def publish_text(text: str) -> str:
    """Publish a text."""
    return f"Published: {text}"


@tool(writes={"published": "text"})
def apology_text(query: str) -> str:
    """Apologize for a query."""
    return f"Sorry, no short answer about {query}."


VERDICT = {
    "fits": {"type": "bool", "means": "the draft makes one claim in under 300 words"},
    "action": {"type": "text", "one of": ["publish", "shorten"], "means": "what should happen next"},
}
"""The judge's answer form."""


def worked_example(verdicts: Iterator[str], **kwargs: Any) -> MAS:
    """The worked example without its module and web_fetch, the judge answering from ``verdicts``.

    The writer drafts, ``count`` counts its words and the judge decides: back to
    the writer while the draft does not fit (at most 3 visits), else publish or
    apologize by the verdict's action.
    """
    llm = MockLLM(lambda p: "A short draft about cats." if p.startswith("Answer") else next(verdicts))
    writer = Agent(
        "writer",
        "Answer {query} in a short draft. The last verdict, empty at first: {verdict}",
        output="draft",
        inputs=["query", "verdict"],
    )
    judge = Agent(
        "judge",
        "Judge this draft: {draft}\nIts word counts: {counts}",
        output="verdict",
        inputs=["draft", "counts"],
        output_type="json",
        schema=VERDICT,
    )
    return MAS(
        [writer, judge],
        [
            ("writer", "count"),
            ("count", "judge"),
            Edge("judge", "writer", EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
            Edge("judge", "publish", EdgeKind.SWITCH, switch="verdict.action"),
            Edge("judge", "apology", EdgeKind.ELSE),
        ],
        code_nodes=[
            Code("count", "count_words"),
            Code("publish", "publish_text", {"text": "draft"}),
            Code("apology", "apology_text"),
        ],
        state={"published": {"merge": "last"}},
        llm=llm,
        tools=[count_words, publish_text, apology_text],
        **kwargs,
    )


# ------------------------------------------------- the deck: for each chains

DECK = """\
version: 2
inputs: {plan: json, house: text}
outputs: {deck: json, notes: text}
agents:
- {name: drafter, model: writer-model, for each: {item: page, in: plan.slides, at once: 4, key: id, item budget: {cost_usd: 0.05}},
   reads: {slide: page, rules: house, verdict: verdict}, writes: {draft: text}}
- {name: checker, model: judge-model, prompt: judge, for each: page in plan.slides, reads: {planned: page, shown: measure},
   writes: {verdict: json}, schema: {fits: {type: bool, means: the page makes its claim at a glance}, fix: {type: text}}}
- {name: captioner, model: writer-model, for each: {item: figure, in: plan.figures, key: id},
   reads: {figure: figure, rules: house}, writes: {caption: text}}
- {name: figure_checker, model: judge-model, prompt: judge, for each: figure in plan.figures,
   reads: {planned: figure, shown: caption}, writes: {figure_verdict: json}, schema: {fits: {type: bool}, fix: {type: text}}}
- {name: noter, model: writer-model, reads: [plan], writes: {notes: text}}
code_nodes:
- {name: render, uses: render_page, for each: page in plan.slides, retry: {outage: 1}}
- {name: assemble, uses: assemble_deck}
edges:
- drafter -> render -> checker
- 'checker -> drafter': {when: not verdict.fits, max_visits: 2}
- 'checker -> assemble': else
- captioner -> figure_checker -> assemble
callables:
  render_page: {kind: python, import: 'tests.conftest:render_page', reads: {draft: text}, writes: {measure: json}}
  assemble_deck: {kind: python, import: 'tests.conftest:assemble_deck', reads: {draft: json, verdict: json, caption: json},
                  writes: {deck: json}}
prompts:
  drafter: "Draft slide {slide} in this house style: {rules}\\nThe last verdict, empty at first: {verdict}"
  judge: "Does the page make the claim of its plan?\\nPlan: {planned}\\nAs drawn: {shown}"
  captioner: "Caption figure {figure} in this house style: {rules}"
  noter: "Write speaker notes for this plan: {plan}"
"""  # noqa: E501
"""The deck of two chains: a page is drafted, rendered and checked, redrawn once while it does not fit; figures are
captioned and checked beside it; ``assemble`` joins both, and END joins it with ``noter``."""

PLAN = {
    "slides": [
        {"id": name, "claim": f"the {name} slide"} for name in ("intro", "method", "result", "limits", "outlook")
    ],
    "figures": [{"id": "f1", "shows": "the setup"}, {"id": "f2", "shows": "the curve"}],
}
"""The deck's plan: five slides and two figures. ``limits`` never fits; ``intro`` is drawn slowly."""

LONG_PAGE = "A long page with far too many words on it for one slide."
"""A first draft: more than eight words, so the rendered page overflows."""


@tool(writes={"measure": "json"})
def render_page(draft: str) -> dict[str, Any]:
    """Render a page and measure it: its words, and whether they overflow it."""
    words = len(draft.split())
    return {"words": words, "overflow": words > 8}


@tool(writes={"deck": "json"})
def assemble_deck(draft: list[str], verdict: list[Any], caption: list[str]) -> dict[str, Any]:
    """Put the pages and the captions together, with whether each page fits."""
    return {"pages": draft, "captions": caption, "fits": [item["fits"] for item in verdict]}


def draft(prompt: str) -> str:
    """The writer: a long first draft, a short one once a verdict came back; ``limits`` stays long."""
    if prompt.startswith("Draft slide"):
        if '"intro"' in prompt:
            time.sleep(0.05)  # the first item finishes last
        time.sleep(0.01)
        redrawn = prompt.rstrip().endswith("}") and '"limits"' not in prompt
        return "Short page." if redrawn else LONG_PAGE
    if prompt.startswith("Caption figure"):
        return "A caption."
    return "Notes."


def judge(prompt: str) -> str:
    """The judge: a page fits unless its render overflows."""
    if '"overflow": true' in prompt:
        return '{"fits": false, "fix": "cut it"}'
    return '{"fits": true, "fix": ""}'


def deck_llms() -> dict[str, MockLLM]:
    """The deck's scripted models, by the names its agents bind; each call costs 0.01."""
    return {"writer-model": MockLLM(draft, meta=True), "judge-model": MockLLM(judge, meta=True)}


def deck() -> MAS:
    """The deck of :data:`DECK` on :func:`deck_llms`."""
    return MAS.from_dict(yaml.safe_load(DECK), llms=dict(deck_llms()), tools=[render_page, assemble_deck])


# ------------------------------------------------------------ media: pictures

PNG = b"\x89PNG-a"
"""The bytes of a tiny picture: six of them, enough for a ref."""


@pytest.fixture
def png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny picture file, ``a.png``; the store (``XMAS_STORE``) lives beside it for the test."""
    monkeypatch.setenv("XMAS_STORE", str(tmp_path / "store"))
    path = tmp_path / "a.png"
    path.write_bytes(PNG)
    return path


class SeeingLLM(MockLLM):
    """A model that accepts images: it records every list of content parts it is sent in ``parts``.

    It reads each picture's bytes, as a request would, and answers parts with
    ``answer`` (a text, or a function of the parts), or with the next of
    ``answers``, as :class:`MockLLM` answers text.
    """

    accepts: tuple[str, ...] = ("text", "image")

    def __init__(self, answer: str | Callable[[list[Any]], str] = "ok", *, answers: list[str] | None = None) -> None:
        """Answer every call with ``answer``, or with the next of ``answers``."""
        super().__init__(answers=list(answers) if answers is not None else None, meta=True)
        self.answer = answer
        self.parts: list[list[Any]] = []

    def generate_parts_with_meta(self, parts: Sequence[Any], system_prompt: str = "") -> tuple[str, CallMeta]:
        """Record the parts, read their bytes and answer them."""
        self.parts.append(list(parts))
        for part in parts:
            if isinstance(part, Ref):
                part.read_bytes()
        if self.answers:
            return self.next_answer(""), META
        return (self.answer if isinstance(self.answer, str) else self.answer(list(parts))), META


class PaintingLLM(MockLLM):
    """A model that produces images: each image call answers ``text`` and ``count`` pictures of :data:`PNG`."""

    produces = ("image",)

    def __init__(self, text: str = "drawn", count: int = 1) -> None:
        """Answer every image call with ``text`` and ``count`` pictures."""
        super().__init__(meta=True)
        self.text, self.count = text, count
        self.asked: list[Any] = []

    def generate_image_with_meta(
        self, prompt: str | Sequence[Any], system_prompt: str = ""
    ) -> tuple[str, list[tuple[bytes, str]], CallMeta]:
        """Record the prompt and paint."""
        self.asked.append(prompt)
        return self.text, [(PNG + bytes([index]), "image/png") for index in range(self.count)], META


@tool(writes={"slide_png": "list of image"})
def render_pages(plan: dict[str, Any]) -> list[Image]:
    """Draw each slide of a plan as a picture in the store."""
    return [
        Ref.from_bytes(f"page {slide['id']}".encode(), "image/png", f"{slide['id']}.png") for slide in plan["slides"]
    ]


MEDIA = """\
version: 2
inputs: {plan: json, house: document}
outputs: {review: text}
agents:
- {name: reviewer, model: vision-model, reads: {pages: slide_png, rules: house_rules, plan: plan}, writes: {review: text}}
code_nodes:
- {name: house_text, uses: document_text, reads: {source: house}, writes: {content: house_rules}}
- {name: render, uses: render_pages}
edges:
- house_text -> render -> reviewer
models:
  vision-model: {accepts: [image]}
callables:
  document_text: {kind: python, import: 'xmas.convert:document_text', reads: {source: document}, writes: {content: text}}
  render_pages: {kind: python, import: 'tests.conftest:render_pages', reads: {plan: json}, writes: {slide_png: list of image}}
prompts:
  reviewer: "Review the drawn deck against its plan and the house rules.\\nPlan: {plan}\\nRules: {rules}\\nPages: {pages}"
"""  # noqa: E501
"""A reviewer that reads the drawn pages as pictures, on a model that accepts images, and the house rules as text."""


# ------------------------------------------------------------- live gateway


def gateway_configured() -> bool:
    """Whether the environment has what a live gateway call needs."""
    return bool(os.environ.get("XMAS_BASE_URL")) and bool(os.environ.get("XMAS_API_KEY"))


requires_gateway = pytest.mark.skipif(
    not gateway_configured(),
    reason="set XMAS_BASE_URL and XMAS_API_KEY to run live gateway tests",
)
"""Skip marker for tests that need Tom's gateway. CI has no key, so these skip."""
