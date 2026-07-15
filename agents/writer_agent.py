"""WriterAgent: drafts and revises documents, seeded by ResearchAgent output when present."""
from __future__ import annotations

from langchain.agents import AgentState, create_agent
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_groq import ChatGroq
from langgraph.store.base import BaseStore

from state import GlobalState
from tools.writer import draft_document, get_draft, revise_document

_WRITER_TOOLS = [draft_document, get_draft, revise_document]
_MODEL = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

_SYSTEM_PROMPT = (
    "You are WriterAgent, part of SmartDesk. Draft or revise documents "
    "(reports, emails, README files) matching the requested format and "
    "tone. If prior research context (a summary or notes) is present in "
    "the conversation, use it as source material and cite it naturally -- "
    "do not fabricate facts not present in that context or your own "
    "general knowledge. You must NEVER call revise_document until AFTER "
    "you have called get_draft for that doc_id in this same turn -- "
    "revise_document's new_content must be real composed text based on "
    "what get_draft returned, never a placeholder, a description of what "
    "you plan to do, or a guess at content you have not fetched. If you "
    "are unsure whether you already have the current content, call "
    "get_draft again rather than guessing. Call draft_document for a NEW "
    "document and revise_document to change an EXISTING one -- never "
    "both for the same task. Once revise_document or draft_document "
    "succeeds for the requested document, that document is persisted; "
    "do not call the other tool afterward as a second persistence step."
)

_TRANSIENT_ERROR_MARKER = "tool_use_failed"


class WriterState(AgentState):
    """Extends AgentState with user_id, so draft_document/get_draft/
    revise_document can resolve their InjectedState("user_id") argument."""
    user_id: str


def _build_writer_agent(store: BaseStore):
    """Compile the WriterAgent subgraph, bound to the given store."""
    return create_agent(
        model=_MODEL,
        tools=_WRITER_TOOLS,
        system_prompt=_SYSTEM_PROMPT,
        state_schema=WriterState,
        store=store,
        name="writer_agent",
    )


def _build_handoff_context(state: GlobalState) -> list:
    """Explicitly surface ResearchAgent's output for WriterAgent.

    Returns a list containing a single context message if research
    output exists, else an empty list. This is the handoff boundary:
    WriterAgent must not rely on ambient message history to receive
    ResearchAgent's summary. Uses SystemMessage rather than
    HumanMessage since this is orchestration context injected by the
    graph, not an instruction from the user.
    """
    summary = state.get("summary", "")
    if not summary:
        return []
    return [
        SystemMessage(
            content=(
                "Research context from ResearchAgent (use as source "
                f"material, do not fabricate beyond it):\n\n{summary}"
            )
        )
    ]


def writer_agent_node(state: GlobalState, store: BaseStore, max_retries: int = 2) -> dict:
    """Run WriterAgent and return a GlobalState patch.

    Explicitly injects ResearchAgent's summary (via state["summary"])
    into the message history before invocation, so the handoff does
    not depend on message-list continuity from the caller. Writes its
    own output to `writer_output` rather than `summary`, so it never
    clobbers ResearchAgent's summary in shared state -- the supervisor
    is responsible for combining both into a final response. Retries
    on a transient malformed tool-call generation (a known Groq/Llama
    issue). Any other tool or model failure -- including a
    ToolException that ToolNode swallows internally into an error
    ToolMessage instead of raising -- is caught and reported via
    `error`, so a sub-agent failure never crashes the graph.
    """
    agent = _build_writer_agent(store)
    handoff_context = _build_handoff_context(state)
    input_messages = state["messages"] + handoff_context
    result = None

    for attempt in range(max_retries + 1):
        try:
            result = agent.invoke(
                {"messages": input_messages, "user_id": state["user_id"]}
            )
            break
        except ToolException as exc:
            return {"error": f"WriterAgent tool failure: {exc}"}
        except Exception as exc:
            if _TRANSIENT_ERROR_MARKER in str(exc) and attempt < max_retries:
                continue
            return {"error": f"WriterAgent failed: {type(exc).__name__}: {exc}"}

    new_messages = result["messages"][len(input_messages):]

    tool_error = next(
        (m.content for m in new_messages if isinstance(m, ToolMessage) and m.status == "error"),
        None,
    )
    if tool_error:
        return {"error": f"WriterAgent tool failure: {tool_error}"}

    writer_output = new_messages[-1].content if new_messages else state.get("writer_output", "")
    return {"messages": new_messages, "writer_output": writer_output, "error": None}