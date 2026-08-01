"""FunctionGemma routing constants and parser re-export."""

from .parser import parse_functiongemma_output
from .schemas import FG_TOOL_SCHEMAS

DEVELOPER_MESSAGE = "You are a model that can do function calling with the following functions"

__all__ = ["DEVELOPER_MESSAGE", "FG_TOOL_SCHEMAS", "parse_functiongemma_output"]
