"""Persistence tests for notes shared across conversation threads."""
from memory.store import SmartDeskStore, build_persistent_store


def test_notes_survive_store_restart(tmp_path):
    path = str(tmp_path / "memory.sqlite")
    with build_persistent_store(path) as backend:
        SmartDeskStore(backend).save_note(
            "user-1", "Checkpointing", "Verified findings", ["langgraph"]
        )

    with build_persistent_store(path) as backend:
        notes = SmartDeskStore(backend).list_notes("user-1", "langgraph")

    assert len(notes) == 1
    assert notes[0]["content"] == "Verified findings"
