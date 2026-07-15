"""Manual live check for ResearchAgent -- real GROQ + TAVILY calls, no mocks.
Run directly: python test_live.py
Requires GROQ_API_KEY and TAVILY_API_KEY set in .env.
"""
from dotenv import load_dotenv
load_dotenv()

import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from agents.research_agent import research_agent_node
from state import GlobalState


def make_state(text: str) -> GlobalState:
    return GlobalState(
        messages=[HumanMessage(content=text)],
        user_id="test-user-1",
        active_agent=None,
        task_queue=[],
        completed_tasks=[],
        memories=[],
        summary="",
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


def case_real_search() -> None:
    """Prompts a real research request; expects at least web_search to run,
    and a note to be persisted as a result."""
    print("=== Case 1: real web search ===")
    store = InMemoryStore()
    state = make_state(
        "Search the web for today's top headline on LangChain's official "
        "blog, save it as a note, and tell me the title and URL."
    )
    patch = research_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("summary:", patch.get("summary"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"

    tool_called = any(isinstance(m, ToolMessage) for m in patch.get("messages", []))
    assert tool_called, "Expected at least one real tool call"

    saved = list(store.search(("notes", "test-user-1")))
    assert saved, "Expected save_note to have written a note"
    print(">>> PASS: real search + save_note confirmed end-to-end.")




if __name__ == "__main__":
    cases = [("Case 1", case_real_search)]
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