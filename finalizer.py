"""Supervisor-owned final response."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import model
from observability import log
from state import GlobalState
from supervisor import is_greeting

logger = log("finalizer")
def finalize(state: GlobalState) -> dict:
    """Create the user-visible answer from direct specialist results or chat context."""
    latest = next((str(m.content) for m in reversed(state["messages"])
                   if isinstance(m, HumanMessage)), "")
    document = state.get("current_document") or {}
    if state.get("active_agent") == "show_document":
        if document.get("content"):
            logger.info("returned canonical document | doc_id=%s", document.get("doc_id"))
            content = str(document["content"])
        else:
            logger.info("no current document available")
            content = "There is no current document to show."
        return {"messages": [AIMessage(content)], "agent_outputs": None}
    if state.get("active_agent") == "chat" and is_greeting(latest):
        logger.info("static greeting response")
        return {"messages": [AIMessage("Ask me to research, plan, or write something.")],
                "agent_outputs": None}

    outputs = [x for x in state.get("agent_outputs", [])
               if x.get("turn_id") == state["turn_id"] and x.get("status") != "failed"]
    if (any(x.get("agent") == "writer" for x in outputs)
            and document.get("turn_id") == state["turn_id"] and document.get("content")):
        logger.info("returned newly persisted document | doc_id=%s", document.get("doc_id"))
        return {"messages": [AIMessage(str(document["content"]))], "agent_outputs": None}
    if state.get("active_agent") == "research_task" and outputs:
        by_agent = {output["agent"]: output["content"] for output in outputs}
        answer = "\n\n".join(
            f"{agent.title()}:\n{by_agent[agent]}"
            for agent in ("research", "task") if agent in by_agent)
        logger.info("completed | deterministic joined output")
        return {"messages": [AIMessage(answer)], "agent_outputs": None}
    if outputs and state.get("active_agent") != "research_task":
        logger.info("completed | agent output passthrough")
        return {"messages": [AIMessage(outputs[-1]["content"])], "agent_outputs": None}

    prompt = ("Answer the user's latest request directly and concisely. Return plain text, never "
              "JSON. Use relevant user memories when helpful. If specialist results exist, "
              "combine only those successful results. "
              f"Results: {outputs}. Summary: {state.get('summary', '')}. "
              f"Relevant user memories: {state.get('memories', [])}.")
    try:
        answer = str(model().invoke([SystemMessage(prompt), *state["messages"]]).content).strip()
    except Exception as exc:
        logger.warning("synthesis failed | error=%s: %s", type(exc).__name__, exc)
        answer = ""
    if not answer:
        answer = outputs[-1]["content"] if outputs else "I could not complete that response."
    logger.info("completed")
    return {"messages": [AIMessage(answer)], "agent_outputs": None}
