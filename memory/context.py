"""Layer 1: in-context conversation memory."""
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from models import model
from state import GlobalState


def trim_history(state: GlobalState) -> dict:
    """Summarise old history after 12 user turns and keep the latest six."""
    messages = state["messages"]
    indexes = [i for i, message in enumerate(messages)
               if isinstance(message, HumanMessage)]
    if len(indexes) <= 12:
        return {}
    old = messages[:indexes[-6]]
    transcript = "\n".join(str(message.content) for message in old)
    previous = state.get("summary", "")
    prompt = (f"Previous summary:\n{previous}\n\nMerge in these newer facts, "
              f"decisions, and preferences concisely:\n{transcript}")
    try: summary = str(model().invoke([SystemMessage(prompt)]).content)
    except Exception: summary = transcript[-2000:]
    return {"summary": summary,
            "messages": [RemoveMessage(id=m.id) for m in old if m.id]}
