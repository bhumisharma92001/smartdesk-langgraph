import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.store.memory import InMemoryStore  

from tools.notes import list_notes, save_note  


def test_save_note_then_list_notes_roundtrip():
    store = InMemoryStore()
    save_note.func("T", "C", ["research"], user_id="u1", store=store)

    result = list_notes.func(None, user_id="u1", store=store)

    assert len(result) == 1
    assert result[0]["title"] == "T"


def test_list_notes_filters_by_tag():
    store = InMemoryStore()
    save_note.func("A", "x", ["work"], user_id="u1", store=store)
    save_note.func("B", "y", ["personal"], user_id="u1", store=store)

    result = list_notes.func("work", user_id="u1", store=store)

    assert len(result) == 1
    assert result[0]["title"] == "A"


def test_notes_are_isolated_per_user():
    store = InMemoryStore()
    save_note.func("Mine", "secret", [], user_id="u1", store=store)

    result = list_notes.func(None, user_id="u2", store=store)

    assert result == []