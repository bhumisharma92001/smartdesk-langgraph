from langchain_core.tools import ToolException


class SmartDeskError(Exception):
    """Base class for every custom exception raised anywhere in SmartDesk."""


class ToolError(SmartDeskError, ToolException):
    """Base class for tool errors.

    Also inherits ToolException, satisfying the spec's rule that all
    tool errors must be caught and re-raised as ToolException.
    """


class ToolInputError(ToolError):
    """Raised when a tool is given input it cannot process."""


class ToolAuthError(ToolError):
    """Raised when a required API key or credential is missing."""


class ToolExecutionError(ToolError):
    """Raised when a tool's underlying operation (API call, parsing) fails."""
    