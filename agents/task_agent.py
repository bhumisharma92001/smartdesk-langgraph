"""TaskAgent: breaks a high-level goal into ordered steps, executes them, and tracks completion."""
from __future__ import annotations

import json
from functools import lru_cache

from langchain.agents import AgentState, create_agent
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.store.base import BaseStore

from agents.common import run_agent
from memory.context import build_conversation_summary_context
from models import reasoning_model
from prompts.task_prompts import TASK_SYSTEM_PROMPT
from state import GlobalState
from tools.notes import save_note
from tools.tasks import create_task, get_task, mark_step_done

_TASK_TOOLS = [create_task, get_task, mark_step_done, save_note]

class TaskState(AgentState):
    """Task agent state exposed to injected tools."""
    user_id: str


@lru_cache(maxsize=8)
def _build_task_agent(store: BaseStore):
    """Build the TaskAgent graph for a store."""
    return create_agent(
        model=reasoning_model(),
        tools=_TASK_TOOLS,
        system_prompt=TASK_SYSTEM_PROMPT,
        state_schema=TaskState,
        store=store,
        name="task_agent",
    )


def _task_updates(messages: list) -> tuple[list[dict], list[dict]]:
    """Extract current tasks and tasks completed by this invocation."""
    tasks: dict[str, dict] = {}
    completed: dict[str, dict] = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        if message.name not in {"create_task", "get_task", "mark_step_done"}:
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        if isinstance(content, dict) and content.get("task_id"):
            tasks[content["task_id"]] = content
            if message.name == "mark_step_done" and content.get("status") == "completed":
                completed[content["task_id"]] = content
    return list(tasks.values()), list(completed.values())


def _task_context(state: GlobalState) -> list[SystemMessage]:
    """Expose persisted active tasks so short continuation requests are unambiguous."""
    tasks = state.get("task_queue", [])
    if not tasks:
        return []
    return [SystemMessage(content=(
        "Persisted active tasks (source of truth). Use the exact persisted task_id. "
        "For 'done' load the task with get_task, then mark exactly its next unfinished "
        "step. For 'next' or 'continue', load and report the next unfinished step without "
        "marking it done:\n"
        f"{json.dumps(tasks)}"
    ))]


def task_agent_node(state: GlobalState, store: BaseStore, max_retries: int = 1) -> dict:
    """Run TaskAgent and synchronize task state from structured results."""
    agent = _build_task_agent(store)
    context = [
        *build_conversation_summary_context(state),
        *_task_context(state),
    ]
    run = run_agent(
        agent, state, name="TaskAgent", context=context, max_retries=max_retries,
        keep_last=10,
    )
    if run.error:
        return {"error": run.error}

    tasks, completed = _task_updates(run.messages)
    patch = {
        "messages": run.messages,
        "task_output": run.output or state.get("task_output", ""),
    }
    if tasks:
        patch["task_queue"] = tasks
    if completed:
        patch["completed_tasks"] = completed
    return patch
