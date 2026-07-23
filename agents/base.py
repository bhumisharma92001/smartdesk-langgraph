"""Small shared agent runner."""
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.store.base import BaseStore
from contracts import AgentReport, cited_urls
from models import model
from observability import log
from state import GlobalState
logger = log("agent")


class SmartDeskAgentState(AgentState):
    """Agent state needed by injected tools."""
    user_id: str


def build(store: BaseStore, tools: list, prompt: str):
    """Compile a specialist subgraph for one supervisor graph."""
    return create_agent(model(), tools, system_prompt=prompt,
                        state_schema=SmartDeskAgentState, store=store,
                        middleware=[ToolRetryMiddleware(
                            max_retries=0, retry_on=(ToolException,),
                            on_failure=lambda exc: str(exc))])


def run(state: GlobalState, agent, name: str) -> tuple[dict, list[ToolMessage]]:
    """Safely invoke one compiled specialist subgraph."""
    logger.info("%s started", name)
    try:
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
        patch = {"messages": [output], "agent_outputs": [report.model_dump(mode="json")]}
        return patch, tools_used
    except Exception as exc:
        logger.exception("%s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}, []
