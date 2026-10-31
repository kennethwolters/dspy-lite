import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from dspy.clients.disk_serialization import DeserializationError
from dspy.clients.sqlite_cache import SqliteCache


class JsonCodec:
    def dumps(self, value):
        return json.dumps(value).encode()

    def loads(self, data):
        try:
            return json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DeserializationError("Invalid entry") from exc


def make_cache(directory, size_limit=None):
    return SqliteCache(directory, size_limit, JsonCodec())


def write_many(directory, prefix):
    cache = make_cache(directory)
    for i in range(30):
        assert cache.set(f"{prefix}-{i}", i)


def test_reopen_and_operations(tmp_path):
    cache = make_cache(tmp_path)
    assert cache.database == tmp_path / "json-v1" / "cache.sqlite3"
    assert cache.directory == str(tmp_path)
    assert cache.get("missing", 42) == 42
    assert cache.set("a", {"value": [1, 2]})
    assert "a" in cache and len(cache) == 1
    cache.close()
    cache = make_cache(tmp_path)
    assert cache.get("a") == {"value": [1, 2]}
    assert cache.delete("a") and not cache.delete("a")
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0


def test_lru_and_limits(tmp_path):
    # Entry cost is UTF-8 key bytes + serialized bytes + 64 metadata bytes.
    cache = make_cache(tmp_path, 134)
    assert cache.set("a", 1)
    assert cache.set("b", 2)
    assert cache.get("a") == 1
    assert cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3
    assert not cache.set("huge", "x" * 200)
    assert len(cache) == 2
    assert not cache.set("a", "x" * 200)
    assert cache.get("a") == 1
    zero = make_cache(tmp_path / "zero", 0)
    assert not zero.set("key", 1)
    assert len(zero) == 0


def test_legacy_ignored_and_temp_directory(tmp_path):
    legacy = tmp_path / "cache.db"
    legacy.write_bytes(b"legacy data")
    cache = make_cache(tmp_path)
    assert len(cache) == 0
    cache.set("key", "value")
    cache.clear()
    cache.close()
    assert legacy.read_bytes() == b"legacy data"
    temp = make_cache(None)
    assert temp.set("key", "value")
    temp.close()
    assert temp.get("key") == "value"


def test_threads_and_processes(tmp_path):
    cache = make_cache(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: write_many(tmp_path, f"thread{i}"), range(4)))
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=write_many, args=(tmp_path, f"process{i}")) for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert len(cache) == 240


def test_corrupt_row_propagates_and_can_be_deleted(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("bad", 1)
    with sqlite3.connect(cache.database) as connection:
        connection.execute("UPDATE entries SET value = ? WHERE key = ?", (b"not json", "bad"))
    with pytest.raises(DeserializationError):
        cache.get("bad")
    assert cache.delete("bad")


def test_corrupt_database_raises_fixed_error(tmp_path):
    cache = make_cache(tmp_path)
    cache.database.write_bytes(b"not a sqlite database")
    with pytest.raises(DeserializationError, match=r"^SQLite cache database error$"):
        len(cache)
