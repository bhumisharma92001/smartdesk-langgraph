"""ResearchAgent: searches the web, fetches pages, and saves findings as notes."""
from __future__ import annotations

from langchain.agents import AgentState, create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langchain_groq import ChatGroq
from langgraph.store.base import BaseStore
import traceback
from state import GlobalState
from tools.fetch_page import fetch_page
from tools.notes import save_note
from tools.search import web_search

_RESEARCH_TOOLS = [web_search, fetch_page, save_note]
_MODEL = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

_SYSTEM_PROMPT = (
    "You are ResearchAgent, part of SmartDesk. Given a user request, search "
    "the web, fetch full page content when a snippet isn't enough, and "
    "produce a concise structured summary. Try at most 2 search queries -- "
    "after that, save_note with the best information you found so far and "
    "stop, even if it isn't a perfect match. Never finish without calling "
    "save_note at least once. Cite sources by URL in your summary."
)

_TRANSIENT_ERROR_MARKER = "tool_use_failed"


class ResearchState(AgentState):
    """Extends AgentState with user_id, so save_note/list_notes can resolve
    their InjectedState("user_id") argument."""
    user_id: str


def _build_research_agent(store: BaseStore):
    """Compile the ResearchAgent subgraph, bound to the given store."""
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return create_agent(
        model=model,
        tools=_RESEARCH_TOOLS,
        system_prompt=_SYSTEM_PROMPT,
        state_schema=ResearchState,
        store=store,
        name="research_agent",
    )


def research_agent_node(state: GlobalState, store: BaseStore, max_retries: int = 2) -> dict:
    """Run ResearchAgent and return a GlobalState patch.

    Retries on a transient malformed tool-call generation (a known
    Groq/Llama issue). Any other tool or model failure -- including a
    ToolException that ToolNode swallows internally into an error
    ToolMessage instead of raising -- is caught and reported via `error`,
    so a sub-agent failure never crashes the graph.
    """
    agent = _build_research_agent(store)
    result = None

    for attempt in range(max_retries + 1):
        try:
            result = agent.invoke(
                {"messages": state["messages"], "user_id": state["user_id"]}
            )
            break
        except ToolException as exc:
            return {"error": f"ResearchAgent tool failure: {exc}"}
        except Exception as exc:
            traceback.print_exc()   # <-- add this
            return {
                "error": f"ResearchAgent failed: {type(exc).__name__}: {exc}"
            }

    new_messages = result["messages"][len(state["messages"]):]

    tool_error = next(
        (m.content for m in new_messages if isinstance(m, ToolMessage) and m.status == "error"),
        None,
    )
    if tool_error:
        return {"error": f"ResearchAgent tool failure: {tool_error}"}

    summary = new_messages[-1].content if new_messages else state.get("summary", "")
    return {"messages": new_messages, "summary": summary, "error": None}