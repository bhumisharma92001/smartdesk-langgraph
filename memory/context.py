"""In-context memory: summarizes and trims conversation history after 12 turns."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from models import fast_model
from state import GlobalState
from utils.logger import get_logger

logger = get_logger(__name__)

_SUMMARY_AFTER_TURNS = 12
_KEEP_RECENT_TURNS = 6

_SUMMARY_PROMPT = (
    "Summarize the conversation history below. Preserve every concrete fact: "
    "names, numbers, decisions, preferences, and anything the user explicitly "
    "asked to remember. Do not replace specific details with vague generalities. "
    "Output only the summary, using dense bullet points when they preserve more "
    "detail than prose."
)


def _count_human_turns(messages: list) -> int:
    """Count user turns, independent of intervening tool messages."""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _split_at_turn_boundary(messages: list, keep_last_n_turns: int) -> tuple[list, list]:
    """Split at a user-turn boundary so tool exchanges stay intact."""
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= keep_last_n_turns:
        return [], messages
    boundary = human_indices[-keep_last_n_turns]
    return messages[:boundary], messages[boundary:]


def _summarize(older_messages: list, existing_summary: str) -> str:
    """Fold older messages into the existing summary."""
    transcript = "\n".join(
        f"{m.__class__.__name__}: {m.content}"
        for m in older_messages
        if getattr(m, "content", None)
    )
    prior = f"Existing summary so far:\n{existing_summary}\n\n" if existing_summary else ""
    prompt = f"{_SUMMARY_PROMPT}\n\n{prior}Conversation to fold in:\n{transcript}"
    response = fast_model().invoke([SystemMessage(content=prompt)])
    return str(response.content or "")


def summarize_history_node(state: GlobalState) -> dict:
    """After 12 user turns, summarize older history and retain six turns."""
    messages = state["messages"]
    if _count_human_turns(messages) <= _SUMMARY_AFTER_TURNS:
        return {}
    logger.info("Conversation history reached summarization threshold")

    older, _ = _split_at_turn_boundary(messages, _KEEP_RECENT_TURNS)
    if not older:
        return {}

    try:
        new_summary = _summarize(older, state.get("summary", ""))
    except Exception as exc:
        logger.warning("History summarization failed: %s", exc)
        return {"error": f"History summarization failed: {type(exc).__name__}: {exc}"}
    removals = [RemoveMessage(id=m.id) for m in older if m.id is not None]

    return {
        "summary": new_summary,
        "messages": removals,
    }


def build_conversation_summary_context(state: GlobalState) -> list:
    """Expose the running conversation summary to an agent."""
    summary = state.get("summary", "")
    if not summary:
        return []
    return [
        SystemMessage(content=f"Earlier conversation summary:\n{summary}")
    ]
