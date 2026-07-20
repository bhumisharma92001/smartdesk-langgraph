from langgraph.graph import END, START, StateGraph

from state import GlobalState
from dotenv import load_dotenv
load_dotenv() 

def passthrough_node(state: GlobalState) -> dict:
    """Echo the latest message back; verifies GlobalState + add_messages wiring."""
    last = state["messages"][-1].content if state["messages"] else ""
    return {"messages": [{"role": "assistant", "content": f"passthrough received: {last!r}"}], "error": None}


def build_smoke_graph():
    """Compile the single-node smoke-test graph."""
    builder = StateGraph(GlobalState)
    builder.add_node("passthrough", passthrough_node)
    builder.add_edge(START, "passthrough")
    builder.add_edge("passthrough", END)
    return builder.compile()