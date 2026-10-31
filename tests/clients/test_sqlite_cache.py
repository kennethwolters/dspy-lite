import json
import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

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
    assert cache.get("a") is None
    zero = make_cache(tmp_path / "zero", 0)
    assert not zero.set("key", 1)
    assert len(zero) == 0


def test_failed_replacement_cannot_restore_old_value(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("key", "old")
    with pytest.raises(TypeError):
        cache.set("key", object())
    assert make_cache(tmp_path).get("key") is None


def test_lowered_limit_applies_to_existing_instances(tmp_path):
    original = make_cache(tmp_path, 1000)
    original.set("key", "x" * 500)
    limited = make_cache(tmp_path, 0)
    assert limited.get("key") is None
    assert original.get("key") is None
    assert not original.set("new", "value")
    assert len(limited) == 0
    expanded = make_cache(tmp_path, 1000)
    assert original.set("new", "value")
    assert expanded.get("new") == "value"


@pytest.mark.parametrize("target", ["database", "namespace", "wal"])
def test_symlinks_cannot_redirect_cache_writes(tmp_path, target):
    outside = tmp_path / "outside"
    outside.mkdir()
    database = outside / "other.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE untouched (value TEXT)")
    before = database.read_bytes()
    cache_root = tmp_path / "cache"
    namespace = cache_root / "json-v1"
    namespace.mkdir(mode=0o700, parents=True)
    if target == "namespace":
        namespace.rmdir()
        namespace.symlink_to(outside, target_is_directory=True)
    else:
        (namespace / ("cache.sqlite3" if target == "database" else "cache.sqlite3-wal")).symlink_to(database)
    with pytest.raises(ValueError, match="private|symlink"):
        make_cache(cache_root)
    assert database.read_bytes() == before


def test_hard_link_cannot_redirect_cache_writes(tmp_path):
    other = tmp_path / "other.sqlite3"
    with sqlite3.connect(other) as connection:
        connection.execute("CREATE TABLE untouched (value TEXT)")
    before = other.read_bytes()
    root = tmp_path / "cache"
    namespace = root / "json-v1"
    namespace.mkdir(mode=0o700, parents=True)
    os.link(other, namespace / "cache.sqlite3")
    with pytest.raises(ValueError, match="hard links"):
        make_cache(root)
    assert other.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory permissions")
def test_shared_writable_paths_are_refused(tmp_path):
    root = tmp_path / "cache"
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    with pytest.raises(ValueError, match="private"):
        make_cache(root)


def test_links_are_refused_after_initialization(tmp_path):
    cache = make_cache(tmp_path / "cache")
    other = tmp_path / "other.sqlite3"
    with sqlite3.connect(other) as connection:
        connection.execute("CREATE TABLE untouched (value TEXT)")
    before = other.read_bytes()
    cache.database.unlink()
    cache.database.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        cache.set("key", "value")
    assert other.read_bytes() == before


def test_concurrent_eviction_respects_shared_budget(tmp_path):
    def write(index):
        cache = make_cache(tmp_path, 256)
        for item in range(15):
            cache.set(f"{index}-{item}", item)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    cache = make_cache(tmp_path, 256)
    with sqlite3.connect(cache.database) as connection:
        assert connection.execute("SELECT SUM(cost) FROM entries").fetchone()[0] <= 256
    assert 0 < len(cache) < 60


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


def test_unlinked_regular_stat_snapshot_is_not_a_hard_link(tmp_path):
    cache = make_cache(tmp_path)
    cache.set("key", "value")
    original = type(cache.database).lstat

    def snapshot(path):
        result = original(path)
        if path == cache.database:
            fields = list(result)
            fields[3] = 0
            return os.stat_result(fields)
        return result

    with patch.object(type(cache.database), "lstat", snapshot):
        assert cache.get("key") == "value"


def test_corrupt_database_raises_fixed_error(tmp_path):
    cache = make_cache(tmp_path)
    cache.database.write_bytes(b"not a sqlite database")
    with pytest.raises(DeserializationError, match=r"^SQLite cache database error$"):
        len(cache)
