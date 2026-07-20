"""WriterAgent: drafts and revises documents, seeded by ResearchAgent output when present."""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import AgentState, create_agent
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore

from agents.common import run_agent
from memory.context import build_conversation_summary_context
from models import reasoning_model
from prompts.writer_prompts import WRITER_SYSTEM_PROMPT
from state import GlobalState
from tools.writer import draft_document, get_draft, revise_document

_WRITER_TOOLS = [draft_document, get_draft, revise_document]

class WriterState(AgentState):
    """Writer agent state exposed to injected tools."""
    user_id: str


@lru_cache(maxsize=8)
def _build_writer_agent(store: BaseStore):
    """Build the WriterAgent graph for a store."""
    return create_agent(
        model=reasoning_model(0.3),
        tools=_WRITER_TOOLS,
        system_prompt=WRITER_SYSTEM_PROMPT,
        state_schema=WriterState,
        store=store,
        name="writer_agent",
    )


def _build_handoff_context(state: GlobalState) -> list:
    """Expose completed Research and Task work to WriterAgent."""
    context = []
    if summary := state.get("research_summary", ""):
        context.append(SystemMessage(content=(
            "Verified ResearchAgent context; use only these findings and cited sources:\n\n"
            f"{summary}"
        )))
    if task_output := state.get("task_output", ""):
        context.append(SystemMessage(content=f"TaskAgent project plan:\n\n{task_output}"))
    return context


def writer_agent_node(state: GlobalState, store: BaseStore, max_retries: int = 1) -> dict:
    """Run WriterAgent and preserve ResearchAgent output."""
    agent = _build_writer_agent(store)
    context = [
        *build_conversation_summary_context(state),
        *_build_handoff_context(state),
    ]
    run = run_agent(
        agent,
        state,
        name="WriterAgent",
        context=context,
        excluded_ai_content={
            state.get("research_summary", ""),
            state.get("task_output", ""),
        },
        max_retries=max_retries,
        keep_last=10,
    )
    if run.error:
        return {"error": run.error}
    return {
        "messages": run.messages,
        "writer_output": run.output or state.get("writer_output", ""),
    }
