"""Small in-process guards for chat requests."""
from __future__ import annotations

from collections import deque
import threading
import time


class RequestController:
    """Limit per-user request frequency and serialize each conversation thread."""

    def __init__(self, limit: int = 10, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._thread_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def allow(self, user_id: str) -> bool:
        now = time.monotonic()
        with self._guard:
            requests = self._requests.setdefault(user_id, deque())
            cutoff = now - self._window
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self._limit:
                return False
            requests.append(now)
            return True

    def thread_lock(self, thread_id: str) -> threading.Lock:
        with self._guard:
            return self._thread_locks.setdefault(thread_id, threading.Lock())
