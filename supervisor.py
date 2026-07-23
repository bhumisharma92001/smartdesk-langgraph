"""Hybrid LLM supervisor with deterministic routing safety checks."""
import re
from typing import Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from models import model
from observability import log
from state import GlobalState

logger = log("supervisor")
Workflow = Literal["chat", "show_document", "research", "task", "writer", "research_writer",
                   "task_writer", "research_task", "research_task_writer"]
GREETINGS = {"hi", "hy", "hye", "hello", "helo", "hey", "thanks",
             "thank you", "good morning", "good evening"}
PLANNER_PROMPT = (
    "Return the smallest workflow and any durable user memories in one structured response. "
    "The workflow must be exactly one of: chat, show_document, research, "
    "task, writer, research_writer, task_writer, research_task, or research_task_writer. "
    "Use task for creating, organizing, listing, or updating tasks/checklists, including nuanced "
    "requests such as 'help me get organized' or 'break this goal down for me'. Use writer for "
    "drafting or revising text. Only include research when the user explicitly asks to research, "
    "search, browse, look up, find online, or provide externally sourced/cited information. "
    "Words such as review, content, test, website, deploy, launch, production, report, or source "
    "code do not by themselves require research. Task plus an email/document is task_writer; "
    "explicit external research plus task management is research_task; add writer only when the "
    "user also asks for drafted content. A request to revise or edit an existing draft routes to "
    "writer; a request merely to share/show the current WriterAgent draft routes to show_document. "
    "show_document is exclusively for the current document, never for notes. Any request to list, "
    "show, retrieve, or identify a note—including one containing a note UUID—must route to task so "
    "TaskAgent can perform an exact stored-note lookup. Saving research "
    "findings/results as a note is part of ResearchAgent's own workflow and does not require "
    "TaskAgent. Include task only when the user also asks to create, track, update, or manage a "
    "task/checklist. Route saving, listing, showing, retrieving, or identifying notes to task; do not call "
    "research merely to read saved notes. Never add an agent merely because it might be useful. "
    "For memories, extract only explicit, durable facts about the user that will help in future "
    "conversations: identity, preferences, stable constraints, relationships, or long-term goals. "
    "Represent each memory with type, key, and value fields. "
    "Resolve natural phrasing such as 'call me Bhoomi' or 'always use a formal tone'. Do not save "
    "the current request, research facts, assistant output, temporary details, secrets, or inferred "
    "facts. Preserve the user's meaning in a short standalone sentence; otherwise return an empty list."
)


class MemoryFact(BaseModel):
    """One explicit durable fact extracted from the user's message."""

    type: Literal["identity", "preference", "constraint", "relationship", "long_term_goal"]
    key: str
    value: str

    def text(self) -> str:
        """Render a compact standalone fact for semantic storage."""
        key = self.key.strip().replace("_", " ")
        value = self.value.strip()
        if self.type == "identity" and key.casefold() == "name":
            return f"The user's name is {value}."
        return f"The user's {key} is {value}."


class Plan(BaseModel):
    """Validated workflow and memories selected in one LLM call."""

    workflow: Workflow = Field(description=(
        "Use task for every note operation, including showing a note by UUID. "
        "Use show_document only for the current WriterAgent document."
    ))
    memories: list[MemoryFact] = Field(default_factory=list)


def is_greeting(text: str) -> bool:
    """Recognize common greeting spellings without an LLM call."""
    value = text.lower().strip(" !.,")
    return value in GREETINGS or re.sub(r"(.)\1+", r"\1", value) in GREETINGS


def plan_turn(text: str, has_active_tasks: bool = False) -> Plan:
    """Plan routing and durable memories in one structured model call."""
    context = "The user currently has active tasks." if has_active_tasks else \
              "The user currently has no active tasks."
    decision = model().with_structured_output(Plan).invoke(
        [SystemMessage(content=f"{PLANNER_PROMPT}\n{context}"), HumanMessage(content=text)])
    logger.info("LLM planner selected | workflow=%s", decision.workflow)
    return decision


def plan_workflow(text: str, has_active_tasks: bool = False) -> Workflow:
    """Compatibility helper returning only the selected workflow."""
    return plan_turn(text, has_active_tasks).workflow


def supervise(state: GlobalState) -> dict:
    """Plan the complete workflow and record supervisor ownership."""
    text = next(str(message.content) for message in reversed(state["messages"])
                if isinstance(message, HumanMessage))
    try:
        decision = (Plan(workflow="chat") if is_greeting(text) else
                    plan_turn(text, bool(state.get("task_queue"))))
    except Exception as exc:
        logger.exception("planner failed")
        return {"active_agent": None, "turn_id": str(uuid4()),
                "error": f"Planner failed: {type(exc).__name__}", "agent_outputs": None,
                "new_memories": []}
    workflow = decision.workflow
    turn_id = str(uuid4())
    logger.info("planned | workflow=%s turn_id=%s", workflow, turn_id)
    return {"active_agent": workflow, "turn_id": turn_id, "error": None,
            "agent_outputs": None,
            "new_memories": list(dict.fromkeys(
                fact.text() for fact in decision.memories if fact.value.strip()))}


def selected(state: GlobalState) -> str:
    """Return the supervisor's recorded workflow."""
    return str(state["active_agent"])


def planner_status(state: GlobalState) -> Literal["memory", "error"]:
    """Stop immediately when the supervisor could not produce a workflow."""
    return "error" if state.get("error") else "memory"


def after_agent(state: GlobalState, handoff: str) -> Literal["error", "writer", "final"]:
    """Continue after a successful agent, optionally handing its result to WriterAgent."""
    if state.get("error"): return "error"
    successful = [entry for entry in state.get("agent_outputs", [])
                  if entry.get("turn_id") == state.get("turn_id")
                  and entry.get("status") != "failed" and entry.get("content")]
    if not successful: return "error"
    return "writer" if handoff and state.get("active_agent") == handoff else "final"


def after_join(state: GlobalState) -> Literal["error", "writer", "final"]:
    """Route joined parallel work to WriterAgent, final synthesis, or safe failure."""
    if state.get("error"): return "error"
    return "writer" if state.get("active_agent") == "research_task_writer" else "final"
