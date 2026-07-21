"""Supervisor-owned final synthesis and fact extraction."""
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from models import model
from observability import log
from state import GlobalState
from supervisor import is_greeting
logger = log("finalizer")
SHARE_DOCUMENT = re.compile(
    r"\b(share|show|display|paste|print)\b(?:\s+(?:the|it|that|full|complete|report|document|"
    r"draft|content|here))*\b|\b(full|complete)\s+(report|document|draft|text)\b", re.I)
NOTE_REFERENCE = re.compile(r"\s*(?:note\s*)?#?(\d+)\s*", re.I)
PERSONAL_FACT_SIGNAL = re.compile(
    r"\b(my name is|i am|i like|i love|i prefer|my favou?rite|remember that|"
    r"mera naam|mujhe .{0,30} pasand)\b", re.I)


class FinalResponse(BaseModel):
    answer: str
    facts: list[str] = Field(default_factory=list,
        description="Explicit first-person facts stated BY THE USER about themselves only — "
                    "never facts about external topics, research findings, or general knowledge.")


def fallback_facts(state: GlobalState) -> list[str]:
    """Extract explicit first-person facts if structured output is unavailable."""
    text = next((str(m.content) for m in reversed(state["messages"])
                 if isinstance(m, HumanMessage)), "")
    pattern = (r"\b(?:my name is|i am|i like|i prefer|my favorite \w+ is|remember that)"
               r"\s+[^.!?;]{1,100}")
    return [match.group(0).strip() for match in re.finditer(pattern, text, re.I)]


def finalize(state: GlobalState) -> dict:
    """Create the only user-visible answer after all monitored work."""
    latest = next((str(m.content) for m in reversed(state["messages"])
                   if isinstance(m, HumanMessage)), "")
    note_match = NOTE_REFERENCE.fullmatch(latest)
    notes = state.get("last_note_list", [])
    if state.get("active_agent") == "chat" and note_match:
        index = int(note_match.group(1)) - 1
        if 0 <= index < len(notes):
            note = notes[index]
            logger.info("returned canonical note | note_id=%s", note.get("id"))
            return {"messages": [AIMessage(f"{note.get('title', 'Note')}\n\n"
                                           f"{note.get('content', '')}")],
                    "new_memories": [], "agent_outputs": None, "monitor_log": None}
    document = state.get("current_document") or {}
    if (state.get("active_agent") == "chat" and document.get("content") and
            SHARE_DOCUMENT.search(latest)):
        logger.info("returned canonical document | doc_id=%s", document.get("doc_id"))
        return {"messages": [AIMessage(str(document["content"]))], "new_memories": [],
                "agent_outputs": None, "monitor_log": None}
    if state.get("active_agent") == "chat" and is_greeting(latest):
        logger.info("static greeting response")
        return {"messages": [AIMessage("Ask me to research, plan, or write something.")],
                "new_memories": fallback_facts(state), "agent_outputs": None,
                "monitor_log": None}
    outputs = [x for x in state.get("agent_outputs", [])
               if x.get("turn_id") == state["turn_id"]]
    reviews = [x for x in state.get("monitor_log", [])
               if x.get("turn_id") == state["turn_id"]]
    approved = {x["agent"] for x in reviews if x.get("approved")}
    rejected = [x for x in reviews if not x.get("approved")]
    if approved: outputs = [x for x in outputs if x.get("agent") in approved]
    if ("writer" in approved and document.get("turn_id") == state["turn_id"] and
            document.get("content")):
        logger.info("returned newly persisted document | doc_id=%s", document.get("doc_id"))
        return {"messages": [AIMessage(str(document["content"]))],
                "new_memories": fallback_facts(state), "agent_outputs": None,
                "monitor_log": None}
    if outputs and state.get("active_agent") != "research_task" and not rejected:
        logger.info("completed | monitored output passthrough")
        return {"messages": [AIMessage(outputs[-1]["content"])],
                "new_memories": fallback_facts(state), "agent_outputs": None,
                "monitor_log": None}
    prompt = ("Return one coherent answer using only approved results. Briefly mention any "
              "failed specialist without exposing internal errors, while still completing all "
              f"possible work. Approved results: {outputs}. Reviews: {reviews}. "
              f"Summary: {state.get('summary', '')}.")
    raw = parsed = None
    try:
        result = model().with_structured_output(FinalResponse, include_raw=True).invoke(
            [SystemMessage(prompt), *state["messages"]])
        parsed, raw = result.get("parsed"), result.get("raw")
    except Exception as exc: logger.warning("structured synthesis failed | error=%s: %s", type(exc).__name__, exc)
    answer = parsed.answer if parsed else str(getattr(raw, "content", ""))
    if not answer: answer = outputs[-1]["content"] if outputs else "I could not complete that response."
    facts = (parsed.facts if parsed else fallback_facts(state)) if PERSONAL_FACT_SIGNAL.search(
        latest) else []
    logger.info("completed | structured=%s facts=%d", bool(parsed), len(facts))
    return {"messages": [AIMessage(answer)], "new_memories": facts, "agent_outputs": None,
            "monitor_log": None}
