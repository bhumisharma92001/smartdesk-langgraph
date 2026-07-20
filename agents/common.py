"""Shared execution helpers for SmartDesk sub-agents."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException

from state import GlobalState
from utils.logger import get_logger

_TRANSIENT_ERROR = "tool_use_failed"
_AGENT_RECURSION_LIMIT = 8
logger = get_logger(__name__)


def _preview(value, limit: int = 500) -> str:
    return " ".join(str(value).split())[:limit]


class _AgentTrace(BaseCallbackHandler):
    """Log concise model and tool progress without dumping full conversations."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        latest = messages[0][-1].content if messages and messages[0] else ""
        logger.info("%s model_start input=%s", self.agent_name, _preview(latest))

    def on_llm_end(self, response, **kwargs) -> None:
        message = response.generations[0][0].message
        tools = [call["name"] for call in getattr(message, "tool_calls", [])]
        if tools:
            logger.info("%s model_end tool_calls=%s", self.agent_name, tools)
        else:
            logger.info("%s model_end output=%s", self.agent_name, _preview(message.content))

    def on_llm_error(self, error, **kwargs) -> None:
        logger.error("%s model_error=%s", self.agent_name, error)

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        logger.info(
            "%s tool_start name=%s input=%s",
            self.agent_name, serialized.get("name", "tool"), _preview(input_str),
        )

    def on_tool_end(self, output, **kwargs) -> None:
        logger.info("%s tool_end output=%s", self.agent_name, _preview(output))

    def on_tool_error(self, error, **kwargs) -> None:
        logger.error("%s tool_error=%s", self.agent_name, error)


@dataclass(frozen=True)
class AgentRun:
    messages: list[BaseMessage]
    error: str | None = None

    @property
    def output(self) -> str:
        return str(self.messages[-1].content) if self.messages else ""


def _unrecovered_tool_error(messages: list[BaseMessage]) -> str | None:
    """Return a tool error not followed by a successful retry."""
    last_error_message: ToolMessage | None = None
    last_error = -1
    last_final = -1
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            if message.status == "error":
                last_error = index
                last_error_message = message
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            last_final = index
    if last_error_message is None or last_final > last_error:
        return None
    return f"{last_error_message.name or 'tool'}: {last_error_message.content}"


def compact_history(
    messages: list[BaseMessage],
    excluded_ai_content: set[str] | None = None,
    keep_last: int = 24,
) -> list[BaseMessage]:
    """Keep conversational messages while dropping internal tool traces.

    keep_last controls how many recent conversational messages are kept.
    Specialist agents (research/task/writer) pass a smaller value since
    they need the current request plus explicit handoff context, not
    deep chat history.
    """
    excluded = excluded_ai_content or set()
    history = [
        message
        for message in messages
        if (
            not isinstance(message, (ToolMessage, SystemMessage))
            and not (
                isinstance(message, AIMessage)
                and (message.tool_calls or str(message.content) in excluded)
            )
        )
    ]
    return history[-keep_last:]


def run_agent(
    agent,  # which agent to run
    state: GlobalState,
    *,
    name: str,  # eg: research_agent
    context: list[BaseMessage] | None = None,  # any extra info
    excluded_ai_content: set[str] | None = None,  # skip old msg
    max_retries: int = 2,
    keep_last: int = 24,
) -> AgentRun:
    """Run a sub-agent safely, with retries and error handling.

    Combines extra context + cleaned-up chat history, runs the agent,
    and checks if any tool call failed. If the model sends a badly
    formatted tool call, retries up to `max_retries` times. Any other
    error stops immediately, no retry.

    Args:
        agent: the sub-agent to run (ResearchAgent/TaskAgent/WriterAgent).
        state: current conversation state (messages, user_id).
        name: agent's name, used in error messages.
        context: extra background messages for this turn.
        excluded_ai_content: old messages to skip, to avoid duplicates.
        max_retries: how many times to retry a malformed tool call.
        keep_last: how many recent conversational messages to include.

    Returns:
        AgentRun: new messages produced, and an error (or None if it worked).
    """
    input_messages = [
        *(context or []),
        *compact_history(state["messages"], excluded_ai_content, keep_last),
    ]
    for attempt in range(max_retries + 1):
        try:
            config = {
                "recursion_limit": _AGENT_RECURSION_LIMIT,
                "callbacks": [_AgentTrace(name)],
            }
            result = agent.invoke(
                {"messages": input_messages, "user_id": state["user_id"]},
                config=config,
            )
            messages = result["messages"][len(input_messages):]
            error = _unrecovered_tool_error(messages)
            return AgentRun(
                messages=messages,
                error=f"{name} tool failure: {error}" if error else None,
            )
        except ToolException as exc:
            return AgentRun([], f"{name} tool failure: {exc}")
        except Exception as exc:
            if _TRANSIENT_ERROR in str(exc) and attempt < max_retries:
                input_messages.append(
                    SystemMessage(
                        content=(
                            "Your previous tool call was rejected as malformed. Retry using "
                            "the provider's native structured tool-call format only; do not "
                            "write function-call markup in message text."
                        )
                    )
                )
                continue
            return AgentRun([], f"{name} failed: {type(exc).__name__}: {exc}")
    return AgentRun([], f"{name} failed after {max_retries + 1} attempts")
