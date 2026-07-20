"""Offline tests for durable identity and thread ownership."""
import pytest

from backend.auth import IdentityStore


def test_returning_username_restores_same_user(tmp_path):
    store = IdentityStore(str(tmp_path / "users.sqlite"))
    user_id, token = store.login("Bhoomi", None)
    restored, replacement = store.login("bhoomi", token)
    assert restored == user_id
    assert replacement is None
    store.close()


def test_thread_cannot_be_used_by_another_user(tmp_path):
    store = IdentityStore(str(tmp_path / "users.sqlite"))
    first, _ = store.login("first", None)
    second, _ = store.login("second", None)
    thread_id = store.thread(first, None)
    with pytest.raises(PermissionError):
        store.thread(second, thread_id)
    store.close()
