"""Manual live check for WriterAgent -- real GROQ calls, no mocks.
Run directly: python test_live_writer.py
Requires GROQ_API_KEY set in .env.

Covers three things the code review flagged as risky:
  1. A plain draft (no research context) actually persists via draft_document.
  2. The Research -> Writer handoff: when state["summary"] is set, that
     content shows up in the agent's output, AND the node does not
     clobber state["summary"] (it writes to writer_output instead).
  3. Revision: the agent calls get_draft before revise_document, rather
     than guessing at content it never fetched.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from agents.writer_agent import writer_agent_node
from state import GlobalState


def make_state(text: str, summary: str = "") -> GlobalState:
    return GlobalState(
        messages=[HumanMessage(content=text)],
        user_id="test-user-1",
        active_agent=None,
        task_queue=[],
        completed_tasks=[],
        memories=[],
        summary=summary,
        writer_output="",
        error=None,
    )


def print_trace(messages) -> None:
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [TOOL CALL] {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            tag = "  <ERROR>" if m.status == "error" else ""
            print(f"  [TOOL RESULT{tag}] {m.name}: {str(m.content)[:200]}")
        elif isinstance(m, AIMessage) and m.content:
            print(f"  [AI FINAL] {str(m.content)[:300]}")


def case_plain_draft() -> None:
    """No research context. Expects draft_document to be called and
    writer_output to be populated, with no error."""
    print("=== Case 1: plain draft, no research context ===")
    store = InMemoryStore()
    state = make_state(
        "Write a short formal email to a client named Priya letting her "
        "know the project deadline has moved to next Friday."
    )
    patch = writer_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("writer_output:", patch.get("writer_output"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"

    tool_called = any(
        isinstance(m, ToolMessage) and m.name == "draft_document"
        for m in patch.get("messages", [])
    )
    assert tool_called, "Expected draft_document to be called"

    saved = list(store.search(("drafts", "test-user-1")))
    assert saved, "Expected draft_document to have written a draft"
    print(">>> PASS: plain draft persisted end-to-end.")


def case_handoff_from_research() -> None:
    """Simulates ResearchAgent having already run by pre-populating
    state["summary"]. Expects that content to be used in the draft, AND
    expects the node patch to NOT include a "summary" key (so it can
    never clobber ResearchAgent's summary when merged into GlobalState)."""
    print("=== Case 2: handoff from simulated ResearchAgent summary ===")
    store = InMemoryStore()
    research_summary = (
        "Key finding: LangGraph 0.2 introduced native subgraph support, "
        "letting supervisors invoke compiled StateGraphs as nodes directly. "
        "Source: https://langchain-ai.github.io/langgraph/"
    )
    state = make_state(
        "Draft a short internal report summarizing what we found in our "
        "research, formal tone.",
        summary=research_summary,
    )
    patch = writer_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("writer_output:", patch.get("writer_output"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"

    assert "summary" not in patch, (
        "writer_agent_node must not return a 'summary' key -- doing so "
        "would overwrite ResearchAgent's summary in shared state"
    )

    saved = list(store.search(("drafts", "test-user-1")))
    assert saved, "Expected a draft to be saved"
    draft_content = saved[0].value["content"].lower()
    assert "langgraph" in draft_content or "subgraph" in draft_content, (
        "Expected the research summary content to show up in the draft -- "
        "handoff context may not be reaching the agent"
    )
    print(">>> PASS: research summary was used and not clobbered.")


def case_revision_uses_get_draft() -> None:
    """Pre-seeds a draft directly in the store, then asks for a revision.
    Expects get_draft to be called before revise_document, and expects
    the stored content to actually change."""
    print("=== Case 3: revision calls get_draft before revise_document ===")
    store = InMemoryStore()
    from memory.store import SmartDeskStore

    doc_id = SmartDeskStore(store).save_draft(
        user_id="test-user-1",
        topic="Team update",
        content="Hi team, quick update: things are on track.",
        format="email",
        tone="casual",
    )

    state = make_state(
        f"Revise document {doc_id} to sound more formal and add a "
        "closing line thanking the team."
    )
    patch = writer_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("writer_output:", patch.get("writer_output"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"

    tool_names = [
        m.name for m in patch.get("messages", []) if isinstance(m, ToolMessage)
    ]
    assert "get_draft" in tool_names, (
        "Expected get_draft to be called to fetch existing content before revising"
    )
    assert "revise_document" in tool_names, "Expected revise_document to be called"

    first_get_draft = tool_names.index("get_draft")
    first_revise = tool_names.index("revise_document")
    assert first_get_draft < first_revise, (
        "get_draft was called AFTER revise_document, not before. This means "
        "the model attempted a revision with guessed/placeholder content "
        "before ever fetching the real draft -- even if it later "
        "self-corrected, that first bad call still persisted to the store."
    )
    assert "draft_document" not in tool_names, (
        "draft_document was called during a pure revision task. This "
        "creates an orphaned duplicate document instead of only updating "
        "the existing one via revise_document."
    )

    updated = SmartDeskStore(store).get_draft("test-user-1", doc_id)
    assert updated["content"] != "Hi team, quick update: things are on track.", (
        "Expected the draft content to actually change after revision"
    )
    assert updated["revisions"], "Expected a revision log entry to be recorded"

    all_drafts = list(store.search(("drafts", "test-user-1")))
    assert len(all_drafts) == 1, (
        f"Expected exactly 1 draft in the store after a pure revision task, "
        f"found {len(all_drafts)} -- the model likely created an orphaned "
        f"duplicate document instead of only updating the existing one."
    )
    print(">>> PASS: revision correctly fetched existing content before rewriting.")


if __name__ == "__main__":
    cases = [
        ("Case 1", case_plain_draft),
        ("Case 2", case_handoff_from_research),
        ("Case 3", case_revision_uses_get_draft),
    ]
    results = {}

    for name, fn in cases:
        try:
            fn()
            results[name] = "PASS"
        except AssertionError as exc:
            results[name] = f"FAIL: {exc}"
        except Exception as exc:
            results[name] = f"ERROR: {exc}"
        print()

    print("=== Summary ===")
    for name, status in results.items():
        print(f"{name}: {status}")