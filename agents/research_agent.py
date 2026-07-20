"""ResearchAgent: searches the web and saves findings as notes."""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

from agents.common import run_agent
from memory.context import build_conversation_summary_context
from models import reasoning_model
from prompts.research_prompts import RESEARCH_SYSTEM_PROMPT
from state import GlobalState
from tools.calculator import calculator
from tools.notes import save_note
from tools.search import web_search

_RESEARCH_TOOLS = [web_search, save_note, calculator]

_SUMMARY_CHARS = 1800

class ResearchState(AgentState):
    """Research agent state exposed to injected tools."""
    user_id: str


@lru_cache(maxsize=8)
def _build_research_agent(store: BaseStore):
    """Build the ResearchAgent graph for a store."""
    return create_agent(
        model=reasoning_model(),
        tools=_RESEARCH_TOOLS,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        state_schema=ResearchState,
        store=store,
        name="research_agent",
    )


def _bounded_summary(text: str) -> str:
    """Bound handoff size while retaining source lines whenever possible."""
    if len(text) <= _SUMMARY_CHARS:
        return text
    sources = "\n".join(
        line for line in text.splitlines() if "http://" in line or "https://" in line
    )
    suffix = f"\n\nSources:\n{sources}" if sources else ""
    return text[: max(0, _SUMMARY_CHARS - len(suffix))].rstrip() + suffix


def research_agent_node(state: GlobalState, store: BaseStore, max_retries: int = 2) -> dict:
    """Run ResearchAgent without allowing failures to escape the graph."""
    agent = _build_research_agent(store)
    context = [
        *build_conversation_summary_context(state),
    ]
    run = run_agent(
        agent, state, name="ResearchAgent", context=context, max_retries=max_retries,
        keep_last=6,
    )
    if run.error:
        return {"error": run.error}
    summary = _bounded_summary(run.output or state.get("research_summary", ""))
    messages = run.messages
    if messages and isinstance(messages[-1], AIMessage):
        messages = [*messages[:-1], messages[-1].model_copy(update={"content": summary})]
    return {
        "messages": messages,
        "research_summary": summary,
    }
