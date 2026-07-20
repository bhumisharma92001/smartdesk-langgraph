"""Top-level SmartDesk supervisor graph.

Implements the Supervisor pattern: a lightweight routing node classifies
each request and dispatches to the specialist subgraph(s) that own the
actual multi-step reasoning (Research, Task, Writer). The supervisor
itself is intentionally not an agent with its own tool-calling loop --
classification is a single deterministic decision, and wrapping it in a
ReAct loop would add latency and non-determinism for no benefit. Every
routing decision is recorded to `routing_log` for observability.
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from agents.research_agent import research_agent_node
from agents.task_agent import task_agent_node
from agents.writer_agent import writer_agent_node
from memory.store import get_smartdesk_store
from models import fast_model
from prompts.router_prompts import ROUTER_SYSTEM_PROMPT
from state import GlobalState
from utils.logger import get_logger

logger = get_logger(__name__)

Workflow = Literal[
    "chat", "notes", "research", "writer", "task",
    "research_writer", "research_task", "research_task_writer",
]

_GREETINGS = {"hi", "hello", "hey", "thanks", "thank you"}
_TASK_CONTINUATIONS = {
    "next", "done", "continue", "complete", "mark done", "finished",
}
_TOOL_MARKUP = ("<tool_call", "<arg_key", "<arg_value", "</tool_call")


class Route(BaseModel):
    """Structured supervisor routing decision."""

    workflow: Workflow = Field(description="Smallest workflow that completes the request")


def _latest_human_message(state: GlobalState) -> str:
    """Return the most recent user turn, or an empty string if none exists."""
    return next(
        (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


def _log_route(text: str, workflow: str, *, fallback: bool = False) -> dict[str, object]:
    """Build the state patch for a routing decision, including its audit trail."""
    entry = {"input": text[:200], "workflow": workflow, "fallback": fallback}
    return {
        "active_agent": workflow,
        "error": None,
        "routing_log": [entry],
    }


def _fallback_route(text: str) -> Workflow:
    """Recover from unavailable or malformed router-model output.

    This is intentionally used only after the structured router fails. Notes
    are checked first so phrases such as "search my notes" never become web
    research requests.
    """
    value = text.lower()
    if re.search(r"\b(notes?|saved findings)\b", value):
        return "notes"

    research = bool(re.search(
        r"\b(research|current|latest|today|price|prices|source|sources|verify|web)\b",
        value,
    ))
    writer = bool(re.search(r"\b(write|draft|report|document|email|article)\b", value))
    task = bool(re.search(r"\b(plan|task|steps|checklist|schedule)\b", value))

    if research and task and writer:
        return "research_task_writer"
    if research and writer:
        return "research_writer"
    if research and task:
        return "research_task"
    if research:
        return "research"
    if writer:
        return "writer"
    if task:
        return "task"
    return "chat"


def _parsed_route(result: object) -> Route | None:
    """Read a route from native structured output or its raw JSON fallback."""
    if isinstance(result, Route):
        return result
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("parsed"), Route):
        return result["parsed"]

    raw = result.get("raw")
    content = getattr(raw, "content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        return Route.model_validate_json(content)
    except ValueError:
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if not match:
            return None
        try:
            return Route.model_validate_json(match.group(0))
        except ValueError:
            return None


def supervisor_node(state: GlobalState) -> dict[str, object]:
    """Classify the latest request into a workflow.

    A greeting/thanks is routed without a model call, since that
    classification carries zero ambiguity and paying LLM latency for it
    is pure waste. Every other request goes through the router model --
    keyword heuristics were deliberately removed here because they
    misrouted phrases like "search my notes" (a `notes` request) into
    `research`. A wrong deterministic guess is worse than one extra
    model call.
    """
    latest = _latest_human_message(state)
    normalized = latest.lower().strip(" !.,")

    if normalized in _GREETINGS:
        logger.info("Fast-path routed workflow=chat")
        return _log_route(latest, "chat")

    if state.get("task_queue") and normalized in _TASK_CONTINUATIONS:
        logger.info("Fast-path routed workflow=task (active task continuation)")
        return _log_route(latest, "task")

    try:
        result = fast_model().with_structured_output(
            Route, method="json_schema", include_raw=True,
        ).invoke(
            [("system", ROUTER_SYSTEM_PROMPT), ("user", latest)]
        )
        route = _parsed_route(result)
        if route is None:
            workflow = _fallback_route(latest)
            logger.warning("Router returned no structured output; fallback workflow=%s", workflow)
            return _log_route(latest, workflow, fallback=True)
        logger.info("Router selected workflow=%s", route.workflow)
        return _log_route(latest, route.workflow)
    except Exception as exc:
        workflow = _fallback_route(latest)
        logger.warning("Router failed; fallback workflow=%s: %s", workflow, exc)
        return _log_route(latest, workflow, fallback=True)


def begin_turn_node(_: GlobalState) -> dict[str, object]:
    """Clear error and transient specialist handoffs from the prior turn."""
    return {
        "error": None,
        "research_summary": "",
        "task_output": "",
        "writer_output": "",
    }


def chat_node(state: GlobalState) -> dict[str, object]:
    """Answer greetings and general conversation without exposing specialist tools."""
    try:
        response = fast_model().invoke(state["messages"])
        content = str(response.content or "")
        if response.tool_calls or any(marker in content.lower() for marker in _TOOL_MARKUP):
            logger.warning("Chat model emitted tool-call-like output; discarding it")
            content = (
                "I couldn't safely process that as general chat. Please state whether "
                "you want to research, write, or update a task."
            )
        return {"messages": [AIMessage(content=content)]}
    except Exception as exc:
        logger.error("Chat node failed: %s", exc, exc_info=True)
        return {"error": f"Chat failed: {type(exc).__name__}: {exc}"}


def notes_node(state: GlobalState, store: BaseStore) -> dict[str, object]:
    """List persisted notes directly, bypassing the LLM entirely.

    Notes retrieval has no reasoning step -- it's a lookup. Routing it
    through an agent would spend a model call to decide what a regex
    already decides correctly.
    """
    latest = _latest_human_message(state)
    tag = _extract_tag_filter(latest)
    try:
        notes = get_smartdesk_store(store).list_notes(state["user_id"], tag)
    except Exception as exc:
        logger.error("Notes lookup failed: %s", exc, exc_info=True)
        return {"error": f"Could not list notes: {type(exc).__name__}: {exc}"}

    if not notes:
        text = f"No notes found{f' tagged {tag!r}' if tag else ''}."
    else:
        text = "\n\n".join(
            f"**{note['title']}** (`{note['note_id']}`)\n"
            f"Tags: {', '.join(note['tags']) or 'none'}\n"
            f"{note['content'][:1000]}"
            for note in notes[:20]
        )
    return {"messages": [AIMessage(content=text)]}


def _extract_tag_filter(text: str) -> str | None:
    """Pull an optional tag name out of a natural-language notes request."""
    import re

    match = re.search(r"\btagged?\s+[\"']?([\w-]+)", text, re.IGNORECASE)
    return match.group(1) if match else None


def finalize_node(state: GlobalState) -> dict[str, object]:
    """Combine parallel research + task results without losing persisted metadata.

    Only the `research_task` workflow reaches this node with work still
    to assemble -- every other workflow already produced its final
    message and finalize is a pass-through for them.
    """
    if state.get("active_agent") != "research_task":
        return {}

    research = state.get("research_summary", "")
    task = state.get("task_output", "")
    if not research and not task:
        return {"error": state.get("error") or "Parallel workflow produced no result"}

    tasks = state.get("task_queue", [])
    persisted = "\n\n".join(
        f"**Persisted Task:** {item['title']}\n"
        f"**Task ID:** `{item['task_id']}`\n"
        f"**Status:** {item['status']}\n"
        + "\n".join(
            f"{index}. {step['description']} — {'Done' if step['done'] else 'Pending'}"
            for index, step in enumerate(item["steps"], 1)
        )
        for item in tasks
    )
    sections = [
        f"## Research\n{research}" if research else "## Research\nUnavailable",
        f"## Evaluation Plan\n{task}" if task else "## Evaluation Plan\nUnavailable",
        persisted or "**Persistence warning:** TaskAgent returned no persisted task.",
    ]
    if warning := state.get("error"):
        sections.append(f"**Partial-failure warning:** {warning}")
    return {"messages": [AIMessage(content="\n\n".join(sections))]}


def error_handler_node(state: GlobalState) -> dict[str, object]:
    """Log internal detail and return a stable, safe response."""
    detail = state.get("error") or "Unknown workflow error"
    logger.warning("Turn ended in error_handler: %s", detail)
    return {
        "messages": [AIMessage(content=(
            "I ran into a problem completing that request. Please try again, "
            "or rephrase if the issue continues."
        ))]
    }


def _start_parallel(_: GlobalState) -> dict[str, object]:
    """Create the fan-out point for independent research and task work."""
    return {}


# --- Conditional edge selectors -------------------------------------------

def _route(state: GlobalState) -> str:
    """Dispatch a valid supervisor decision, or divert to error handling."""
    return "error" if state.get("error") else str(state["active_agent"])


def _after_research(state: GlobalState) -> str:
    """Continue a handoff or complete a research-only workflow."""
    if state.get("error"):
        return "error"
    workflow = state.get("active_agent")
    if workflow == "research_task_writer":
        return "task"
    return "writer" if workflow == "research_writer" else "finalize"


def _after_task(state: GlobalState) -> str:
    """Continue the three-agent workflow after its project plan."""
    if state.get("error"):
        return "error"
    return "writer" if state.get("active_agent") == "research_task_writer" else "finalize"


def _complete(state: GlobalState) -> str:
    """Send failed single-agent work to the fallback node."""
    return "error" if state.get("error") else "finalize"


# --- Graph assembly ---------------------------------------------------------

def build_graph(*, store: BaseStore, checkpointer: BaseCheckpointSaver | None = None):
    """Compile SmartDesk: memory maintenance, routing, specialist dispatch,
    parallel fan-out, and centralized error recovery."""
    builder = StateGraph(GlobalState)

    builder.add_node("begin_turn", begin_turn_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("chat", chat_node)
    builder.add_node("notes", notes_node)
    builder.add_node("research", research_agent_node)
    builder.add_node("writer", writer_agent_node)
    builder.add_node("task", task_agent_node)
    builder.add_node("parallel_start", _start_parallel)
    builder.add_node("parallel_research", research_agent_node)
    builder.add_node("parallel_task", task_agent_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("error_handler", error_handler_node)

    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "supervisor")
    builder.add_conditional_edges(
        "supervisor", _route,
        {
            "chat": "chat",
            "notes": "notes",
            "research": "research",
            "writer": "writer",
            "task": "task",
            "research_writer": "research",
            "research_task": "parallel_start",
            "research_task_writer": "research",
            "error": "error_handler",
        },
    )
    builder.add_conditional_edges("chat", _complete, {"finalize": "finalize", "error": "error_handler"})
    builder.add_conditional_edges("notes", _complete, {"finalize": "finalize", "error": "error_handler"})
    builder.add_conditional_edges(
        "research", _after_research,
        {"task": "task", "writer": "writer", "finalize": "finalize", "error": "error_handler"},
    )
    builder.add_conditional_edges("writer", _complete, {"finalize": "finalize", "error": "error_handler"})
    builder.add_conditional_edges(
        "task", _after_task,
        {"writer": "writer", "finalize": "finalize", "error": "error_handler"},
    )
    builder.add_edge("parallel_start", "parallel_research")
    builder.add_edge("parallel_start", "parallel_task")
    builder.add_edge(["parallel_research", "parallel_task"], "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("error_handler", END)

    return builder.compile(store=store, checkpointer=checkpointer)
