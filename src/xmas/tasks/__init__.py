"""Tasks a MAS can be optimized against."""

from xmas.tasks.base import MASTask
from xmas.tasks.gsm8k import GSM8KTask

__all__ = ["GSM8KTask", "MASTask"]
