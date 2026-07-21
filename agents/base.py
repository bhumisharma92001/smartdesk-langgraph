"""Small shared agent runner."""
import json
from langchain.agents import AgentState, create_agent
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.store.base import BaseStore
from contracts import AgentReport, cited_urls
from models import model
from observability import log
from state import GlobalState
logger = log("agent")
_AGENTS: dict[tuple[int, str], object] = {}


class SmartDeskAgentState(AgentState):
    """Agent state needed by injected tools."""
    user_id: str


def build(store: BaseStore, tools: list, prompt: str, name: str):
    """Compile each specialist once per store."""
    key = (id(store), name)
    if key not in _AGENTS:
        _AGENTS[key] = create_agent(model(), tools, system_prompt=prompt,
                                    state_schema=SmartDeskAgentState, store=store)
    return _AGENTS[key]


def run(state: GlobalState, store: BaseStore, tools: list, prompt: str, name: str) -> dict:
    """Compile and safely invoke one specialist subgraph."""
    logger.info("%s started", name)
    try:
        agent = build(store, tools, prompt, name)
        context = [SystemMessage(content=(f"Earlier summary: {state.get('summary', '')}\n"
                  f"Relevant memories: {state.get('memories', [])}\n"
                  f"Active tasks: {state.get('task_queue', [])}\n"
                  f"Current canonical document: {state.get('current_document')}"))]
        result = agent.invoke({"messages": context + state["messages"], "user_id": state["user_id"]},
                              {"recursion_limit": 20})
        tools_used = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        output = result["messages"][-1]
        content = str(output.content).strip()
        errors = [str(m.content) for m in tools_used if getattr(m, "status", "success") == "error"]
        sources = cited_urls(content)
        partial = bool(errors) or (name == "research" and not sources)
        status = "failed" if not content else "partial" if partial else "success"
        report = AgentReport(agent=name, turn_id=state["turn_id"], status=status,
            content=content, sources=sources, tool_errors=errors,
            confidence=0 if status == "failed" else 0.5 if partial else 1)
        logger.info("%s finished | status=%s tools=%d sources=%d errors=%d",
                    name, status, len(tools_used), len(sources), len(errors))
        patch = {"messages": [output], "agent_outputs": [report.model_dump(mode="json")],
                 "tool_results": tools_used}
        for message in reversed(tools_used):
            if message.name != "list_notes": continue
            try: notes = json.loads(str(message.content))
            except (json.JSONDecodeError, TypeError): break
            if isinstance(notes, list): patch["last_note_list"] = notes
            break
        return patch
    except Exception as exc:
        logger.exception("%s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
