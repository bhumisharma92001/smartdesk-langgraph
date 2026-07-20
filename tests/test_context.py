"""Mocked pytest suite for memory/context.py.

Run with: pytest tests/test_context.py -v
No API keys or network required -- _summarize is monkeypatched, so
these test the turn-counting, boundary-splitting, RemoveMessage
construction (verified against the REAL add_messages reducer, not
just checked manually), and injection-helper logic -- not real LLM
summarization quality.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import add_messages

from memory import context as context_module
from memory.context import (
    _count_human_turns,
    _split_at_turn_boundary,
    build_conversation_summary_context,
    summarize_history_node,
)


def _make_turn(n: int) -> list:
    """One turn: HumanMessage + AIMessage, with deterministic ids so
    tests can assert on exactly which messages were removed."""
    return [
        HumanMessage(content=f"question {n}", id=f"human-{n}"),
        AIMessage(content=f"answer {n}", id=f"ai-{n}"),
    ]


def _make_conversation(num_turns: int) -> list:
    messages = []
    for i in range(num_turns):
        messages.extend(_make_turn(i))
    return messages


@pytest.fixture(autouse=True)
def mock_summarize(monkeypatch):
    """Replace the real model call with a deterministic stub, and
    record what it was called with so tests can assert on it."""
    calls = []

    def fake_summarize(older_messages, existing_summary):
        calls.append((older_messages, existing_summary))
        return f"[summary of {len(older_messages)} messages, prior={existing_summary!r}]"

    monkeypatch.setattr(context_module, "_summarize", fake_summarize)
    return calls


# ---- Requested test names --------------------------------------------

def test_no_summary_at_12_turns():
    """12 turns exactly must NOT trigger summarization -- only strictly
    more than 12 should."""
    state = {"messages": _make_conversation(12), "summary": ""}
    patch = summarize_history_node(state)
    assert patch == {}


def test_summarizes_after_12_turns():
    state = {"messages": _make_conversation(13), "summary": ""}
    patch = summarize_history_node(state)
    assert "summary" in patch
    assert patch["summary"] != ""
    assert "messages" in patch
    assert all(isinstance(m, RemoveMessage) for m in patch["messages"])


def test_removes_only_old_messages():
    """15 turns, keep last 6 -> the first 9 turns (18 messages) should
    be exactly the ones removed -- nothing from the retained window."""
    state = {"messages": _make_conversation(15), "summary": ""}
    patch = summarize_history_node(state)

    removed_ids = {m.id for m in patch["messages"]}
    expected_removed_ids = {f"human-{i}" for i in range(9)} | {f"ai-{i}" for i in range(9)}
    assert removed_ids == expected_removed_ids


def test_preserves_latest_6_turns():
    """The most important test: apply the returned RemoveMessage
    objects through the REAL add_messages reducer (not a manual id
    check) and confirm the old messages genuinely disappear while all
    six recent turns remain, with no extras and nothing missing."""
    original = _make_conversation(15)
    state = {"messages": original, "summary": ""}
    patch = summarize_history_node(state)

    merged = add_messages(original, patch["messages"])
    merged_ids = {m.id for m in merged}

    removed_ids = {f"human-{i}" for i in range(9)} | {f"ai-{i}" for i in range(9)}
    recent_ids = {f"human-{i}" for i in range(9, 15)} | {f"ai-{i}" for i in range(9, 15)}

    assert removed_ids.isdisjoint(merged_ids), "Old messages must actually be gone after the reducer applies"
    assert recent_ids <= merged_ids, "All six recent turns must still be present after the reducer applies"
    assert len(merged) == len(recent_ids), "No extra or missing messages after the reducer applies"


def test_folds_existing_summary(mock_summarize):
    state = {
        "messages": _make_conversation(15),
        "summary": "Earlier context: the user asked about pricing.",
    }
    summarize_history_node(state)

    assert len(mock_summarize) == 1
    _, existing_summary_arg = mock_summarize[0]
    assert existing_summary_arg == "Earlier context: the user asked about pricing."


def test_research_summary_is_unchanged():
    state = {
        "messages": _make_conversation(15),
        "summary": "",
        "research_summary": "unrelated ResearchAgent output",
    }
    patch = summarize_history_node(state)
    assert "research_summary" not in patch
    assert "writer_output" not in patch
    assert "task_output" not in patch


def test_builds_summary_context_message():
    state = {"summary": "The user previously asked about refund policy."}
    context = build_conversation_summary_context(state)
    assert len(context) == 1
    assert isinstance(context[0], SystemMessage)
    assert "refund policy" in context[0].content


# ---- Additional coverage (kept from the prior round) -------------------

def test_count_human_turns():
    messages = _make_conversation(5)
    assert _count_human_turns(messages) == 5


def test_split_below_threshold_keeps_everything_as_recent():
    messages = _make_conversation(5)
    older, recent = _split_at_turn_boundary(messages, keep_last_n_turns=12)
    assert older == []
    assert recent == messages


def test_split_above_threshold_never_cuts_a_turn_in_half():
    messages = _make_conversation(15)
    older, recent = _split_at_turn_boundary(messages, keep_last_n_turns=12)
    assert isinstance(recent[0], HumanMessage)
    assert recent[0].id == "human-3"
    assert recent[1].id == "ai-3"


def test_no_trim_under_threshold():
    state = {"messages": _make_conversation(5), "summary": ""}
    patch = summarize_history_node(state)
    assert patch == {}


def test_injection_helper_returns_empty_when_no_summary():
    assert build_conversation_summary_context({"summary": ""}) == []


def test_injection_helper_returns_empty_when_summary_key_missing():
    assert build_conversation_summary_context({}) == []


def test_injection_helper_never_touches_research_summary():
    state = {"summary": "", "research_summary": "should not appear here"}
    assert build_conversation_summary_context(state) == []
