"""Typed supervisor review of every specialist result."""
from functools import partial
from typing import Literal
from pydantic import BaseModel, ValidationError
from contracts import AgentReport
from models import model
from observability import log
from state import GlobalState
logger = log("monitor")


class Review(BaseModel):
    """Structured decision for an ambiguous partial result."""
    approved: bool
    feedback: str


def review(state: GlobalState, agent: str) -> dict:
    """Validate evidence cheaply; use an LLM only for partial results."""
    outputs = [x for x in state.get("agent_outputs", [])
               if x.get("turn_id") == state["turn_id"] and x.get("agent") == agent]
    if not outputs:
        reason = state.get("error") or f"{agent} returned no result"
        logger.error("%s rejected | error=%s", agent, reason)
        return {"error": reason, "monitor_log": [{"turn_id": state["turn_id"],
                "agent": agent, "approved": False, "feedback": str(reason)}]}
    try: report = AgentReport.model_validate(outputs[-1])
    except ValidationError as exc:
        reason = f"Invalid {agent} result: {exc}"
        return {"error": reason, "monitor_log": [{"turn_id": state["turn_id"],
                "agent": agent, "approved": False, "feedback": reason}]}
    if report.status == "failed":
        reason = f"{agent} failed to produce output"
        return {"error": reason, "monitor_log": [{"turn_id": state["turn_id"],
                "agent": agent, "approved": False, "feedback": reason}]}
    if report.status == "success":
        logger.info("%s approved | deterministic contract check", agent)
        return {"monitor_log": [{"turn_id": state["turn_id"], "agent": agent, "approved": True,
                "feedback": "validated contract and evidence"}]}
    prompt = (f"Review this partial {agent} result. Tool errors: {report.tool_errors}; "
              f"sources: {report.sources}; content: {report.content}")
    try: decision = model().with_structured_output(Review).invoke(prompt)
    except Exception: decision = Review(approved=False, feedback="partial result was not reviewable")
    patch = {"monitor_log": [{"turn_id": state["turn_id"], "agent": agent,
                              **decision.model_dump()}]}
    if not decision.approved: patch["error"] = f"Supervisor rejected {agent}: {decision.feedback}"
    logger.info("%s reviewed | approved=%s feedback=%s", agent, decision.approved, decision.feedback)
    return patch


def reviewer(agent: str):
    """Bind an agent name to its supervisor review node."""
    return partial(review, agent=agent)


def after(state: GlobalState, handoff: str) -> Literal["error", "writer", "final"]:
    """Choose failure, WriterAgent handoff, or final synthesis."""
    if state.get("error"): return "error"
    return "writer" if state.get("active_agent") == handoff else "final"


def after_join(state: GlobalState, handoff: str) -> Literal["error", "writer", "final"]:
    """After fan-in, hard-fail only when neither parallel branch was approved."""
    approved = [entry for entry in state.get("monitor_log", [])
                if entry.get("turn_id") == state["turn_id"] and entry.get("approved")]
    if not approved: return "error"
    return "writer" if state.get("active_agent") == handoff else "final"


def done(state: GlobalState) -> Literal["error", "final"]:
    """Choose supervisor final synthesis or graceful failure."""
    return "error" if state.get("error") else "final"
