"""Small SQLite identity and thread-ownership registry for the demo API."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import uuid


class IdentityStore:
    """Create durable users and prevent cross-user thread access."""

    def __init__(self, path: str = "smartdesk_users.sqlite") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
              username TEXT PRIMARY KEY COLLATE NOCASE,
              user_id TEXT UNIQUE NOT NULL,
              token_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS threads(
              thread_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def login(self, username: str, token: str | None) -> tuple[str, str | None]:
        """Register a new username or authenticate an existing browser token."""
        username = username.strip()
        if not username:
            raise ValueError("Username is required")
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, token_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row:
                if not token or not secrets.compare_digest(row[1], self._hash(token)):
                    raise PermissionError("This username already exists on another session")
                return row[0], None
            user_id, new_token = str(uuid.uuid4()), secrets.token_urlsafe(32)
            self._conn.execute(
                "INSERT INTO users VALUES (?, ?, ?)",
                (username, user_id, self._hash(new_token)),
            )
            self._conn.commit()
            return user_id, new_token

    def authorize(self, user_id: str, token: str) -> None:
        """Verify a user token without exposing its stored hash."""
        row = self._conn.execute(
            "SELECT token_hash FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row or not secrets.compare_digest(row[0], self._hash(token)):
            raise PermissionError("Invalid user session")

    def thread(self, user_id: str, thread_id: str | None) -> str:
        """Create a thread or verify that an existing thread belongs to the user."""
        with self._lock:#stops parallel request
            if not thread_id:
                thread_id = str(uuid.uuid4())
                self._conn.execute("INSERT INTO threads VALUES (?, ?)", (thread_id, user_id))
                self._conn.commit()
                return thread_id
            row = self._conn.execute(
                "SELECT user_id FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if not row or row[0] != user_id:
                raise PermissionError("Thread does not belong to this user")
            return thread_id

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
