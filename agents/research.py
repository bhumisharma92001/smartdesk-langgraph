"""ResearchAgent subgraph."""
from langgraph.store.base import BaseStore
from agents.base import run
from state import GlobalState
from tools.notes import save_note
from tools.notes import list_notes
from tools.calculator import calculator
from tools.research import fetch_page, web_search


def run_research(state: GlobalState, store: BaseStore) -> dict:
    """Search, read sources, and optionally save findings."""
    prompt = ("You are ResearchAgent. Use web_search, fetch the most relevant pages, then "
              "summarize their content with cited URLs; save useful findings when requested.")
    patch = run(state, store, [web_search, fetch_page, save_note, list_notes, calculator], prompt,
                "research")
    patch.pop("tool_results", None); return patch
