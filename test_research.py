"""Low-cost live smoke test for Supervisor -> ResearchAgent.

Run directly (not collected by pytest):
    .\\venv\\Scripts\\python.exe test_research.py

Uses real OpenRouter and Tavily credentials from .env. Expected paid work is one
supervisor decision, one web search, and the ResearchAgent's short tool loop.
"""
import re

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from graph import build_graph
from state import GlobalState


SMOKE_QUERY = (
    "Research the current stable Python release. In at most 3 bullets, give its "
    "version and release date, with URLs for python.org and one other credible "
    "source. Do not save a note."
)


def make_state() -> GlobalState:
    return GlobalState(
        messages=[HumanMessage(content=SMOKE_QUERY)],
        user_id="research-smoke-test",
        active_agent=None,
        task_queue=[],
        completed_tasks=[],
        summary="",
        research_summary="",
        writer_output="",
        task_output="",
        routing_log=[],
        error=None,
    )


def main() -> None:
    store = InMemoryStore()
    result = build_graph(store=store).invoke(make_state())

    route = result.get("active_agent")
    trace = result.get("routing_log", [])
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    tool_names = [m.name for m in tool_messages]
    answer = result.get("research_summary", "")
    urls = set(re.findall(r"https?://[^\s)>\]]+", answer))

    print(f"query: {SMOKE_QUERY}")
    print(f"supervisor route: {route}")
    print(f"routing log: {trace}")
    print(f"tools used: {tool_names}")
    print(f"answer: {answer}")

    assert result.get("error") is None, result.get("error")
    assert route == "research", f"Supervisor misrouted request to {route!r}"
    assert trace and trace[-1].get("fallback") is False, "Supervisor used fallback routing"
    assert tool_names.count("web_search") == 1, (
        f"Expected exactly one web_search, got {tool_names}"
    )
    assert "save_note" not in tool_names, "Smoke test should not spend a note-save call"
    assert answer, "ResearchAgent returned no research_summary"
    assert len(urls) >= 2, f"Expected at least two cited URLs, got {sorted(urls)}"
    assert any(isinstance(m, AIMessage) and m.content for m in result["messages"])

    print("\nPASS: supervisor routed to ResearchAgent; one search returned cited research.")


if __name__ == "__main__":
    main()
