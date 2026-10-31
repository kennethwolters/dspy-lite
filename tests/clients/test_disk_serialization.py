"""Real persistent JSON round trips and rejection of executable cache payloads."""

import base64
import os
import pickle
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import orjson
import pydantic
import pytest
from litelm import ModelResponse
from openai.types.responses import Response

from dspy.clients.disk_serialization import DeserializationError, JsonCodec
from dspy.clients.sqlite_cache import SqliteCache


class AllowedModel(pydantic.BaseModel):
    name: str
    value: int


@dataclass
class AllowedDataclass:
    name: str
    value: int


class Outer:
    @dataclass
    class Inner:
        value: int


def roundtrip(tmp_path, value, safe_types=None):
    writer = SqliteCache(tmp_path, 1024 * 1024, JsonCodec(safe_types))
    assert writer.set("key", value)
    writer.close()
    return SqliteCache(tmp_path, 1024 * 1024, JsonCodec(safe_types)).get("key")


@pytest.mark.parametrize("value", [
    {"a": 1, "b": [2, 3]}, [1, "two", 3.0], "hello", 42, (1, 2, 3),
    None, False, b"\x00\xff", {(1, 2): "tuple key"},
])
def test_plain_values_roundtrip(tmp_path, value):
    assert roundtrip(tmp_path, value) == value


@pytest.mark.parametrize("cls", [AllowedModel, AllowedDataclass])
def test_registered_types_roundtrip(tmp_path, cls):
    value = cls(name="sample", value=42)
    assert roundtrip(tmp_path, value, [cls]) == value


def test_nested_class_roundtrip(tmp_path):
    assert roundtrip(tmp_path, Outer.Inner(42), [Outer.Inner]) == Outer.Inner(42)


def test_dataclass_restore_does_not_call_constructor(tmp_path):
    @dataclass
    class Data:
        value: int

        def __post_init__(self):
            if self.value != 1:
                raise RuntimeError("Constructor must not run on cache input")

    codec = JsonCodec([Data])
    payload = orjson.loads(codec.dumps(Data(1)))
    payload["value"][2] = ["dict", [[["value", "value"], ["value", 2]]]]
    assert codec.loads(orjson.dumps(payload)).value == 2


def test_numeric_array_roundtrip(tmp_path):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = roundtrip(tmp_path, value)
    assert result.dtype == value.dtype
    np.testing.assert_array_equal(result, value)


def test_object_arrays_are_rejected():
    with pytest.raises(TypeError):
        JsonCodec().dumps(np.array([object()], dtype=object))
    payload = {"version": 1, "value": ["array", "O", [1], base64.b64encode(b"\0" * 8).decode()]}
    with pytest.raises(DeserializationError):
        JsonCodec().loads(orjson.dumps(payload))


def test_litelm_tool_response_roundtrip(tmp_path):
    response = ModelResponse(id="sample", choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "weather", "arguments": '{"city":"Berlin"}'}}]}}], usage={"total_tokens": 4})
    response._hidden_params = {"response_cost": 0.01}
    result = roundtrip(tmp_path, response)
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.tool_calls[0].function.name == "weather"
    assert result._hidden_params == response._hidden_params
    assert result.usage.total_tokens == 4


def test_text_and_embedding_dicts_roundtrip(tmp_path):
    for value in [{"choices": [{"text": "Hello", "index": 0}]},
                  {"data": [{"embedding": [0.1, 0.2], "index": 0}]}]:
        assert roundtrip(tmp_path, value) == value


def test_responses_api_roundtrip(tmp_path):
    response = Response.model_validate({"id": "resp_1", "created_at": 0.0, "error": None,
        "incomplete_details": None, "instructions": None, "model": "sample", "object": "response",
        "output": [{"id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}]}],
        "metadata": {}, "parallel_tool_calls": False, "temperature": 1.0, "tool_choice": "auto",
        "tools": [], "top_p": 1.0})
    result = roundtrip(tmp_path, response)
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "hello"


def test_unregistered_types_are_rejected_on_write():
    with pytest.raises(TypeError):
        JsonCodec().dumps(AllowedModel(name="sample", value=1))


def test_registrations_are_isolated():
    payload = JsonCodec([AllowedModel]).dumps(AllowedModel(name="sample", value=1))
    with pytest.raises(DeserializationError):
        JsonCodec().loads(payload)


def test_untrusted_type_names_cannot_import_modules():
    payload = {"version": 1, "value": ["provider", "os.system", ["value", "not executable"]]}
    with patch("dspy.clients.disk_serialization.importlib.import_module") as importer:
        with pytest.raises(DeserializationError):
            JsonCodec().loads(orjson.dumps(payload))
        importer.assert_not_called()


def test_pickle_payload_is_never_loaded(tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    payload = pickle.dumps(Payload())
    with patch("pickle.loads", side_effect=AssertionError("Must never unpickle")) as loader:
        with pytest.raises(DeserializationError):
            JsonCodec().loads(payload)
        loader.assert_not_called()
    assert not marker.exists()


@pytest.mark.parametrize("payload", [b"broken", b"{}", b'{"version":2,"value":["value",1]}',
    b'{"version":true,"value":["value",1]}', b'{"version":1,"value":["value",{}]}',
    b'{"version":1,"value":["bytes","not base64!"]}'])
def test_corrupt_or_unknown_entries_are_misses(payload):
    with pytest.raises(DeserializationError):
        JsonCodec().loads(payload)
