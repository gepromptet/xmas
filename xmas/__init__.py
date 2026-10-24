"""xmas: a lightweight framework for building and optimizing multi-agent systems.

xmas is promptolution's abstractions lifted one level. The multi-agent system is
the predictor, and its per-agent prompts, its topology and its per-node model
bindings are components an optimizer can improve. A one-agent system is not an
analogy to a promptolution predictor; it is one.

Note:
    The build reads ``__version__`` from this file (see ``[tool.hatch.version]``
    in ``pyproject.toml``), so keep the assignment a plain string literal.
"""

from xmas.agent import Agent
from xmas.budget import Budget, CancelFlag, cancel_flag, contestant
from xmas.chains import ForEach
from xmas.config import Config, ModelEntry, ModelNameWarning
from xmas.edges import Edge, EdgeKind
from xmas.errors import BadFormat, BudgetExceeded, ErrorKind, RaceLost, RunStatus, Stopped
from xmas.examples import Check, Example
from xmas.graph import END, START
from xmas.health import Health, HealthSettings
from xmas.human import Asker, Human, Pause, Question, ReplayAsker, answer_value
from xmas.llms import APILLM, ScoredSequence
from xmas.mas import MAS
from xmas.media import Audio, Capabilities, Document, File, Image, Ref, Table
from xmas.nodes import Box, Step
from xmas.store import Store as RunStore
from xmas.tasks import GSM8KTask, MASTask
from xmas.trace import AgentRecord, Event, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType
from xmas.verify import VerifyFailed, VerifyReport, VerifyWarning

__version__ = "0.0.0"

__all__ = [
    "APILLM",
    "END",
    "JSON",
    "MAS",
    "START",
    "TEXT",
    "Agent",
    "AgentRecord",
    "Asker",
    "Audio",
    "BadFormat",
    "Box",
    "Budget",
    "BudgetExceeded",
    "CancelFlag",
    "Capabilities",
    "Check",
    "Config",
    "Document",
    "Edge",
    "EdgeKind",
    "ErrorKind",
    "Event",
    "Example",
    "File",
    "ForEach",
    "GSM8KTask",
    "Health",
    "HealthSettings",
    "Human",
    "Image",
    "MASTask",
    "ModelEntry",
    "ModelNameWarning",
    "Pause",
    "Question",
    "RaceLost",
    "Ref",
    "ReplayAsker",
    "RunStatus",
    "RunStore",
    "ScoredSequence",
    "Step",
    "Stopped",
    "Table",
    "TokenUsage",
    "ToolCall",
    "Trace",
    "ValueType",
    "VerifyFailed",
    "VerifyReport",
    "VerifyWarning",
    "__version__",
    "answer_value",
    "cancel_flag",
    "contestant",
]
