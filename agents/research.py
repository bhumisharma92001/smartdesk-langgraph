"""ResearchAgent subgraph."""
from langgraph.store.base import BaseStore
from agents.base import build, run
from state import GlobalState
from tools.notes import save_note, list_notes
from tools.calculator import calculator
from tools.research import fetch_page, web_search


def build_research(store: BaseStore):
    """Compile ResearchAgent and return its supervisor node."""
    prompt = ("You are ResearchAgent. Use web_search, fetch the most relevant pages, then "
              "summarize their content with cited URLs. If the user asks to save the research "
              "as a note, you MUST call save_note and include only its returned note ID. "
              "Never claim that a task or document was created or updated because you do not "
              "have those tools. When the request also asks for task management, return only "
              "research findings and leave the task plan to TaskAgent. Never invent any identifier.")
    agent = build(store, [web_search, fetch_page, save_note, list_notes, calculator], prompt)

    def research(state: GlobalState) -> dict:
        patch, _ = run(state, agent, "research")
        return patch

    return research
