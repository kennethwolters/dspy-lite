"""Persistent JSON cache; the latest configuration sets the shared logical budget."""

import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from dspy.clients.disk_serialization import DeserializationError


class SqliteCache:
    """Connections are per operation; the budget is not a physical file-size cap."""

    _ENTRY_OVERHEAD = 64

    def __init__(self, directory: str | Path | None, size_limit: int | None, codec):
        if size_limit is not None and size_limit < 0:
            raise ValueError("size_limit must be non-negative or None")
        self._temporary_directory = tempfile.TemporaryDirectory() if directory is None else None
        root = Path(self._temporary_directory.name if directory is None else directory)
        if root.is_symlink():
            raise ValueError("Cache paths must be private and cannot be symlinks")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory = str(root.resolve())
        self.database = Path(self.directory) / "json-v1" / "cache.sqlite3"
        self.codec = codec
        self._verify_paths()
        self.database.parent.mkdir(mode=0o700, exist_ok=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(key TEXT PRIMARY KEY, value BLOB NOT NULL, cost INTEGER NOT NULL, accessed INTEGER NOT NULL)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS cache_meta (id INTEGER PRIMARY KEY, sequence INTEGER NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO cache_meta (id, sequence) VALUES (1, 0)")
            connection.execute("CREATE TABLE IF NOT EXISTS cache_settings (id INTEGER PRIMARY KEY, max_cost INTEGER)")
            connection.execute(
                "INSERT INTO cache_settings VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET max_cost=excluded.max_cost",
                (size_limit,),
            )
            self._prune(connection)
            connection.commit()

    def _verify_paths(self):
        root = Path(self.directory)
        if os.name == "posix":
            # Foreign writers must not be able to swap path components between checks and open.
            for path in (root, *root.parents):
                info = path.lstat()
                if (info.st_uid not in (0, os.getuid())
                        or (info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX)):
                    raise ValueError("Cache paths must be private")
        paths = [root, self.database.parent, self.database]
        paths += [Path(str(self.database) + suffix) for suffix in ("-wal", "-shm", "-journal")]
        for path in paths:
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            directory = path in (root, self.database.parent)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Cache paths cannot be symlinks")
            if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
                raise ValueError("Cache path has an unexpected file type")
            if not directory and info.st_nlink > 1:
                raise ValueError("Cache paths cannot have hard links")
            mask = 0o077 if path == self.database.parent else 0o022
            if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & mask):
                raise ValueError("Cache paths must be private")

    @contextmanager
    def _connection(self):
        connection = None
        self._verify_paths()
        try:
            connection = sqlite3.connect(self.database, timeout=10)
            yield connection
        except sqlite3.DatabaseError:
            raise DeserializationError("SQLite cache database error") from None
        finally:
            if connection is not None:
                connection.close()

    @property
    def size_limit(self):
        with self._connection() as connection:
            return connection.execute("SELECT max_cost FROM cache_settings WHERE id = 1").fetchone()[0]

    @staticmethod
    def _touch(connection, key):
        connection.execute("UPDATE cache_meta SET sequence = sequence + 1 WHERE id = 1")
        connection.execute("UPDATE entries SET accessed = (SELECT sequence FROM cache_meta WHERE id = 1) WHERE key = ?", (key,))

    @staticmethod
    def _prune(connection):
        limit = connection.execute("SELECT max_cost FROM cache_settings WHERE id = 1").fetchone()[0]
        if limit is None:
            return
        total = connection.execute("SELECT COALESCE(SUM(cost), 0) FROM entries").fetchone()[0]
        while total > limit:
            oldest = connection.execute("SELECT key, cost FROM entries ORDER BY accessed LIMIT 1").fetchone()
            connection.execute("DELETE FROM entries WHERE key = ?", (oldest[0],))
            total -= oldest[1]

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
        try:
            data = self.codec.dumps(value)
        except Exception:
            self.delete(key)
            raise
        cost = len(key.encode("utf-8")) + len(data) + self._ENTRY_OVERHEAD
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            limit = connection.execute("SELECT max_cost FROM cache_settings WHERE id = 1").fetchone()[0]
            if limit is not None and cost > limit:
                connection.execute("DELETE FROM entries WHERE key = ?", (key,))
                connection.commit()
                return False
            connection.execute("UPDATE cache_meta SET sequence = sequence + 1 WHERE id = 1")
            connection.execute(
                "INSERT OR REPLACE INTO entries (key, value, cost, accessed) "
                "VALUES (?, ?, ?, (SELECT sequence FROM cache_meta WHERE id = 1))",
                (key, data, cost),
            )
            self._prune(connection)
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
