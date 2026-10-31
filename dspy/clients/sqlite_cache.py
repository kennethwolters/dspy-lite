"""Persistent SQLite cache with codec-controlled value serialization.

The size limit bounds logical entry costs, not the physical SQLite file size.
"""

import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from dspy.clients.disk_serialization import DeserializationError


class SqliteCache:
    """A process-safe cache; connections and transactions are per operation."""

    _ENTRY_OVERHEAD = 64

    def __init__(self, directory: str | Path | None, size_limit: int | None, codec):
        if size_limit is not None and size_limit < 0:
            raise ValueError("size_limit must be non-negative or None")
        self._temporary_directory = tempfile.TemporaryDirectory() if directory is None else None
        self.directory = str(self._temporary_directory.name if directory is None else directory)
        self.database = Path(self.directory) / "json-v1" / "cache.sqlite3"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.codec = codec
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(key TEXT PRIMARY KEY, value BLOB NOT NULL, cost INTEGER NOT NULL, accessed INTEGER NOT NULL)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS cache_meta (id INTEGER PRIMARY KEY, sequence INTEGER NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO cache_meta (id, sequence) VALUES (1, 0)")
            connection.commit()

    @contextmanager
    def _connection(self):
        connection = None
        try:
            connection = sqlite3.connect(self.database, timeout=10)
            yield connection
        except sqlite3.DatabaseError:
            raise DeserializationError("SQLite cache database error") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _touch(connection, key):
        connection.execute("UPDATE cache_meta SET sequence = sequence + 1 WHERE id = 1")
        connection.execute("UPDATE entries SET accessed = (SELECT sequence FROM cache_meta WHERE id = 1) WHERE key = ?", (key,))

    def get(self, key, default=None):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
            if row is None:
                return default
            value = self.codec.loads(row[0])
            self._touch(connection, key)
            connection.commit()
            return value

    def set(self, key, value):
        data = self.codec.dumps(value)
        cost = len(key.encode("utf-8")) + len(data) + self._ENTRY_OVERHEAD
        if self.size_limit is not None and cost > self.size_limit:
            return False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE cache_meta SET sequence = sequence + 1 WHERE id = 1")
            connection.execute(
                "INSERT OR REPLACE INTO entries (key, value, cost, accessed) "
                "VALUES (?, ?, ?, (SELECT sequence FROM cache_meta WHERE id = 1))",
                (key, data, cost),
            )
            if self.size_limit is not None:
                total = connection.execute("SELECT COALESCE(SUM(cost), 0) FROM entries").fetchone()[0]
                while total > self.size_limit:
                    oldest = connection.execute("SELECT key, cost FROM entries ORDER BY accessed LIMIT 1").fetchone()
                    connection.execute("DELETE FROM entries WHERE key = ?", (oldest[0],))
                    total -= oldest[1]
            connection.commit()
        return True

    def delete(self, key):
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM entries WHERE key = ?", (key,))
            connection.commit()
            return cursor.rowcount > 0

    def __contains__(self, key):
        with self._connection() as connection:
            return connection.execute("SELECT 1 FROM entries WHERE key = ?", (key,)).fetchone() is not None

    def __len__(self):
        with self._connection() as connection:
            return connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def clear(self):
        with self._connection() as connection:
            connection.execute("DELETE FROM entries")
            connection.commit()

    def close(self):
        """Connections are already closed; subsequent operations remain supported."""
