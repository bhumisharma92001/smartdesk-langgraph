"""Hybrid LLM supervisor with deterministic routing safety checks."""
import re
from typing import Literal, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from models import model
from observability import log
from state import GlobalState

logger = log("supervisor")
Workflow = Literal["chat", "research", "task", "writer", "research_writer",
                   "task_writer", "research_task", "research_task_writer"]
GREETINGS = {"hi", "hy", "hye", "hello", "helo", "hey", "thanks",
             "thank you", "good morning", "good evening"}
PLANNER_PROMPT = (
    "Classify the user's complete intent into exactly one smallest workflow: chat, research, "
    "task, writer, research_writer, task_writer, research_task, or research_task_writer. "
    "Use task for creating, organizing, listing, or updating tasks/checklists, including nuanced "
    "requests such as 'help me get organized' or 'break this goal down for me'. Use writer for "
    "drafting or revising text. Only include research when the user explicitly asks to research, "
    "search, browse, look up, find online, or provide externally sourced/cited information. "
    "Words such as review, content, test, website, deploy, launch, production, report, or source "
    "code do not by themselves require research. Task plus an email/document is task_writer; "
    "explicit external research plus task management is research_task; add writer only when the "
    "user also asks for drafted content. A request to revise or edit an existing draft routes to "
    "writer; a request merely to share/show the existing draft routes to chat. Saving research "
    "findings/results as a note is part of ResearchAgent's own workflow and does not require "
    "TaskAgent. Include task only when the user also asks to create, track, update, or manage a "
    "task/checklist. Never add an agent merely because it might be useful."
)


class Plan(BaseModel):
    """Validated workflow selected by the LLM supervisor."""

    workflow: Workflow


def is_greeting(text: str) -> bool:
    """Recognize common greeting spellings without an LLM call."""
    value = text.lower().strip(" !.,")
    return value in GREETINGS or re.sub(r"(.)\1+", r"\1", value) in GREETINGS


def explicitly_requests_research(text: str) -> bool:
    """Return whether the user explicitly requested external research or citations."""
    return bool(re.search(
        r"\b(research|search|look\s*up|browse|web\s+search|find\s+online)\b|"
        r"\b(cite|provide|include)\b.{0,20}\b(sources?|citations?)\b", text.lower()))


def classify_fallback(text: str) -> Workflow:
    """Classify explicit keywords when the LLM planner is unavailable."""
    value = text.lower()
    research = explicitly_requests_research(value)
    task = (bool(re.search(r"\b(task|plan|steps?|checklist|to[ -]?do|calculate|calculator)\b",
                           value)) or
            bool(re.search(r"\b(create|add|mark|complete)\b.*\b(task|step)\b", value)))
    writer = bool(re.search(r"\b(write|draft|report|email|document|readme|revise|rewrite)\b",
                            value))
    flags = (research, task, writer)
    return {(1, 1, 1): "research_task_writer", (1, 1, 0): "research_task",
            (1, 0, 1): "research_writer", (0, 1, 1): "task_writer",
            (1, 0, 0): "research", (0, 1, 0): "task",
            (0, 0, 1): "writer"}.get(flags, "chat")


def remove_unrequested_research(workflow: Workflow, text: str) -> Workflow:
    """Prevent an LLM from inventing external research the user did not request."""
    if explicitly_requests_research(text): return workflow
    guarded = {"research": "chat", "research_writer": "writer",
               "research_task": "task", "research_task_writer": "task_writer"}.get(
                   workflow, workflow)
    return cast(Workflow, guarded)


def plan_workflow(text: str) -> Workflow:
    """Use intelligent LLM classification with a deterministic failure fallback."""
    try:
        decision = model().with_structured_output(Plan).invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=text)])
        workflow = remove_unrequested_research(decision.workflow, text)
        logger.info("LLM planner selected | raw=%s guarded=%s", decision.workflow, workflow)
        return workflow
    except Exception as exc:
        workflow = classify_fallback(text)
        logger.warning("planner fallback | workflow=%s error=%s: %s", workflow,
                       type(exc).__name__, exc)
        return workflow


def supervise(state: GlobalState) -> dict:
    """Plan the complete workflow and record supervisor ownership."""
    text = next(str(message.content) for message in reversed(state["messages"])
                if isinstance(message, HumanMessage))
    workflow = "chat" if is_greeting(text) else plan_workflow(text)
    turn_id = str(uuid4())
    logger.info("planned | workflow=%s turn_id=%s", workflow, turn_id)
    return {"active_agent": workflow, "turn_id": turn_id, "error": None,
            "agent_outputs": None, "monitor_log": None}


def selected(state: GlobalState) -> str:
    """Return the supervisor's recorded workflow."""
    return str(state["active_agent"])
