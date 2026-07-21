"""Top-level SmartDesk supervisor graph."""
import security  # noqa: F401 - hardening must run before LangGraph imports
from functools import partial
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from agents.research import run_research
from agents.task import run_task
from agents.writer import run_writer
from finalizer import finalize
from memory.logic import load_memories, save_memories
from monitoring import after, after_join, done, reviewer
from observability import log
from state import GlobalState
from supervisor import selected, supervise
logger = log("graph")
def join(state: GlobalState) -> dict:
    """Join parallel branches and clear sibling errors when one branch succeeded."""
    approved = [entry for entry in state.get("monitor_log", [])
                if entry.get("turn_id") == state.get("turn_id") and entry.get("approved")]
    return {"error": None} if approved else {}
def failed(state: GlobalState) -> dict:
    """Turn internal failures into a graceful response."""
    logger.error("workflow stopped | error=%s", state.get("error"))
    return {"messages": [AIMessage("Sorry, I could not safely complete that request.")]}


def build_graph(store, checkpointer=None):
    """Compile supervisor, parallel fan-out/fan-in, and specialist subgraphs."""
    graph = StateGraph(GlobalState)
    nodes = {"memory": load_memories, "supervisor": supervise,
             "research": partial(run_research, store=store), "task": partial(run_task, store=store),
             "writer": partial(run_writer, store=store), "parallel": join, "join": join,
             "parallel_research": partial(run_research, store=store),
             "parallel_task": partial(run_task, store=store), "final": finalize,
             "review_research": reviewer("research"), "review_task": reviewer("task"),
             "review_parallel_research": reviewer("research"),
             "review_parallel_task": reviewer("task"), "review_writer": reviewer("writer"),
             "save": save_memories, "error": failed}
    for name, node in nodes.items(): graph.add_node(name, node)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "memory")
    graph.add_conditional_edges("memory", selected, {"research": "research", "task": "task",
        "writer": "writer", "chat": "final", "research_writer": "research",
        "task_writer": "task", "research_task": "parallel", "research_task_writer": "parallel"})
    graph.add_edge("parallel", "parallel_research"); graph.add_edge("parallel", "parallel_task")
    graph.add_edge("parallel_research", "review_parallel_research")
    graph.add_edge("parallel_task", "review_parallel_task")
    graph.add_edge(["review_parallel_research", "review_parallel_task"], "join")
    graph.add_edge("research", "review_research"); graph.add_edge("task", "review_task")
    graph.add_edge("writer", "review_writer")
    choices = {"error": "error", "writer": "writer", "final": "final"}
    graph.add_conditional_edges("review_research", partial(after, handoff="research_writer"), choices)
    graph.add_conditional_edges("review_task", partial(after, handoff="task_writer"), choices)
    graph.add_conditional_edges("join", partial(after_join, handoff="research_task_writer"), choices)
    graph.add_conditional_edges("review_writer", done, {"error": "error", "final": "final"})
    graph.add_edge("final", "save"); graph.add_edge("save", END); graph.add_edge("error", END)
    return graph.compile(store=store, checkpointer=checkpointer)


if __name__ == "__main__":
    from cli import main
    main()
