"""Top-level SmartDesk supervisor graph."""
import security  # noqa: F401 - hardening must run before LangGraph imports
from functools import partial
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from agents.research import build_research
from agents.task import build_task
from agents.writer import build_writer
from finalizer import finalize
from memory.logic import load_memories, save_memories
from observability import log
from state import GlobalState
from supervisor import after_agent, after_join, planner_status, selected, supervise
logger = log("graph")


def fan_out(_: GlobalState) -> dict:
    """Start parallel specialist branches without changing state."""
    return {}


def join(state: GlobalState) -> dict:
    """Join parallel branches and clear sibling errors when one branch succeeded."""
    successful = [entry for entry in state.get("agent_outputs", [])
                  if entry.get("turn_id") == state.get("turn_id")
                  and entry.get("status") != "failed" and entry.get("content")]
    return {"error": None} if successful else {}
def failed(state: GlobalState) -> dict:
    """Turn internal failures into a graceful response."""
    logger.error("workflow stopped | error=%s", state.get("error"))
    return {"messages": [AIMessage("Sorry, I could not safely complete that request.")]}


def build_graph(store, checkpointer=None):
    """Compile supervisor, parallel fan-out/fan-in, and specialist subgraphs."""
    graph = StateGraph(GlobalState)
    research = build_research(store)
    task = build_task(store)
    writer = build_writer(store)
    nodes = {"memory": load_memories, "supervisor": supervise,
             "research": research, "task": task, "writer": writer,
             "parallel": fan_out, "join": join,
             "parallel_research": research, "parallel_task": task, "final": finalize,
             "save": save_memories, "error": failed}
    for name, node in nodes.items(): graph.add_node(name, node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", planner_status,
                                {"memory": "memory", "error": "error"})
    graph.add_conditional_edges("memory", selected, {"research": "research", "task": "task",
        "writer": "writer", "chat": "final", "show_document": "final",
        "research_writer": "research",
        "task_writer": "task", "research_task": "parallel", "research_task_writer": "parallel"})
    graph.add_edge("parallel", "parallel_research"); graph.add_edge("parallel", "parallel_task")
    graph.add_edge(["parallel_research", "parallel_task"], "join")
    choices = {"error": "error", "writer": "writer", "final": "final"}
    graph.add_conditional_edges("research", partial(after_agent, handoff="research_writer"), choices)
    graph.add_conditional_edges("task", partial(after_agent, handoff="task_writer"), choices)
    graph.add_conditional_edges("join", after_join, choices)
    graph.add_conditional_edges("writer", partial(after_agent, handoff=""), choices)
    graph.add_edge("final", "save"); graph.add_edge("save", END); graph.add_edge("error", END)
    return graph.compile(store=store, checkpointer=checkpointer)


if __name__ == "__main__":
    from cli import main
    main()
