"""Versioned JSON cache payloads with exact, non-executable type registration."""

import base64
import dataclasses
import importlib
import math

import orjson
import pydantic

from dspy.utils.lazy_import import require

np = require("numpy")

# Only these fixed SDK imports may be selected by an untrusted cache entry.
_PROVIDER_TYPES = {
    "litelm-model": ("litelm", "ModelResponse"),
    "litelm-embedding": ("litelm._embedding", "EmbeddingResponse"),
    "litelm-text": ("litelm._text_completion", "TextCompletionResponse"),
    "openai-completion": ("openai.types.completion", "Completion"),
    "openai-chat": ("openai.types.chat", "ChatCompletion"),
    "openai-response": ("openai.types.responses", "Response"),
    "openai-embedding": ("openai.types", "CreateEmbeddingResponse"),
}


class DeserializationError(Exception):
    """A cache entry is incompatible or corrupt and must be treated as a miss."""


def _provider_type(tag):
    module, name = _PROVIDER_TYPES[tag]
    return getattr(importlib.import_module(module), name)


class JsonCodec:
    def __init__(self, safe_types=None):
        from dspy.core.types import LMResponse

        self.safe_types = {f"{LMResponse.__module__}.{LMResponse.__qualname__}": LMResponse}
        for cls in safe_types or []:
            if not isinstance(cls, type):
                raise TypeError("safe_types entries must be types")
            if not (dataclasses.is_dataclass(cls) or issubclass(cls, pydantic.BaseModel)):
                raise TypeError("safe_types must be dataclasses or Pydantic models")
            self.safe_types[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def dumps(self, value):
        return orjson.dumps({"version": 1, "value": self._encode(value)})

    def loads(self, data):
        try:
            document = orjson.loads(data)
            if (set(document) != {"version", "value"} or type(document["version"]) is not int
                    or document["version"] != 1):
                raise ValueError
            return self._decode(document["value"])
        except Exception:
            raise DeserializationError("Invalid or unsupported JSON cache entry") from None

    def _encode(self, value):
        if value is None or type(value) in (str, bool, int, float):
            if isinstance(value, float) and not math.isfinite(value):
                raise TypeError("Non-finite cache values are unsupported")
            return ["value", value]
        if isinstance(value, bytes):
            return ["bytes", base64.b64encode(value).decode("ascii")]
        if isinstance(value, (list, tuple)):
            return ["tuple" if isinstance(value, tuple) else "list", [self._encode(item) for item in value]]
        if isinstance(value, dict):
            return ["dict", [[self._encode(key), self._encode(item)] for key, item in value.items()]]
        cls = type(value)
        name = f"{cls.__module__}.{cls.__qualname__}"
        if self.safe_types.get(name) is cls:
            if isinstance(value, pydantic.BaseModel):
                attributes = value.model_dump(mode="json")
            else:
                attributes = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
            return ["registered", name, self._encode(attributes)]
        if cls.__module__.startswith(("litelm.", "openai.types.")):
            for tag, (module, _) in _PROVIDER_TYPES.items():
                if not cls.__module__.startswith(module.split(".")[0] + ".") or cls is not _provider_type(tag):
                    continue
                if tag == "litelm-model":
                    attributes = {"response": value.model_dump(), "hidden": value._hidden_params}
                elif tag == "litelm-text":
                    attributes = {"response": value._completion.model_dump(mode="json"), "hidden": value._hidden_params}
                elif tag == "litelm-embedding":
                    attributes = value._response.model_dump(mode="json")
                else:
                    attributes = value.model_dump(mode="json")
                return ["provider", tag, self._encode(attributes)]
        if cls.__module__.startswith("numpy"):
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                return ["array", value.dtype.str, list(value.shape), base64.b64encode(value.tobytes()).decode("ascii")]
        raise TypeError("Cache value requires an explicitly registered supported data type")

    def _decode(self, node):
        if not isinstance(node, list) or not node:
            raise ValueError
        kind = node[0]
        if kind == "value" and len(node) == 2:
            value = node[1]
            if value is None or type(value) in (str, bool, int, float):
                return value
            raise ValueError
        if kind == "bytes" and len(node) == 2:
            return base64.b64decode(node[1], validate=True)
        if kind in ("list", "tuple") and len(node) == 2:
            values = [self._decode(item) for item in node[1]]
            return tuple(values) if kind == "tuple" else values
        if kind == "dict" and len(node) == 2:
            return {self._decode(key): self._decode(value) for key, value in node[1]}
        if kind == "registered" and len(node) == 3:
            cls = self.safe_types[node[1]]
            attributes = self._decode(node[2])
            if issubclass(cls, pydantic.BaseModel):
                return cls.model_validate(attributes)
            if not isinstance(attributes, dict) or set(attributes) != {field.name for field in dataclasses.fields(cls)}:
                raise ValueError
            value = object.__new__(cls)
            for key, item in attributes.items():
                object.__setattr__(value, key, item)
            return value
        if kind == "provider" and len(node) == 3:
            cls = _provider_type(node[1])
            attributes = self._decode(node[2])
            if node[1] == "litelm-model":
                value = cls(**attributes["response"])
                value._hidden_params = attributes["hidden"]
                return value
            if node[1] == "litelm-text":
                value = cls(_provider_type("openai-completion").model_validate(attributes["response"]))
                value._hidden_params = attributes["hidden"]
                return value
            if node[1] == "litelm-embedding":
                return cls(_provider_type("openai-embedding").model_validate(attributes))
            return cls.model_validate(attributes)
        if kind == "array" and len(node) == 4:
            dtype = np.dtype(node[1])
            shape = node[2]
            data = base64.b64decode(node[3], validate=True)
            if (dtype.kind not in "biufc" or not isinstance(shape, list)
                    or not all(type(size) is int and size >= 0 for size in shape)
                    or math.prod(shape) * dtype.itemsize != len(data)):
                raise ValueError
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        raise ValueError
