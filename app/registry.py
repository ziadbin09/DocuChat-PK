"""SQLite registry of ingested documents (the vector store holds the chunks)."""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class Registry:
    def __init__(self, path: Path | None):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                       id TEXT PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                       pages INTEGER NOT NULL, chunks INTEGER NOT NULL, bytes INTEGER NOT NULL,
                       created_at TEXT NOT NULL)"""
            )

    def add(self, doc_id: str, name: str, sha256: str, pages: int, chunks: int, size: int) -> dict:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                (doc_id, name, sha256, pages, chunks, size, datetime.now(timezone.utc).isoformat()),
            )
        return self.get(doc_id)

    def get(self, doc_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def by_hash(self, sha256: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0
