"""Native decision lifecycle contracts through the public Predict interface."""

import copy
import json
from typing import Literal

import pytest

import dspy
from dspy.clients.cache import Cache
from dspy.clients.disk_serialization import JsonCodec


ANSWERS = {
    "flag": {"type": "noul", "noul": 0.8},
    "label": {"type": "choice", "choice": "b", "confidence": 0.73, "probabilities": {"a": 0.2, "b": 0.8}},
    "rating": {
        "type": "score",
        "score": 1.5,
        "confidence": 0.61,
        "legend": {"0": "low", "1": "medium", "2": "high"},
        "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
    },
}


def signature(rich=True):
    from dspy.adapters.types.decision import Choice, Noul, Score

    return dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "flag": (Noul if rich else bool, dspy.OutputField(desc="Is it actionable?")),
            "label": (
                Choice[("a", "First"), ("b", "Second")] if rich else Literal["a", "b"],
                dspy.OutputField(desc="Choose a category."),
            ),
            "rating": (Score["low", "medium", "high"], dspy.OutputField(desc="Rate impact.")),
        },
        "Assess the text.",
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    import litelm

    calls = []

    def respond(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return litelm.SystemOneResponse(answers=copy.deepcopy(ANSWERS), usage={"input_tokens": 12})

    async def arespond(**kwargs):
        return respond(**kwargs)

    monkeypatch.setattr(litelm, "systemone", respond)
    monkeypatch.setattr(litelm, "asystemone", arespond)
    monkeypatch.setattr(
        dspy, "cache", Cache(disk_cache_dir=str(tmp_path), enable_disk_cache=True, enable_memory_cache=True)
    )
    return calls


@pytest.mark.parametrize("rich", [True, False])
@pytest.mark.asyncio
async def test_predict_async_history_cache_and_evidence(backend, rich):
    lm = dspy.LM(
        "azure_ai/decision-test",
        model_type="decision",
        api_key="test-secret",
        api_base="https://one.example.test",
        num_retries=0,
    )
    predictor = dspy.Predict(signature(rich))
    predictor.set_lm(lm)
    predictor.demos = [{"text": "example", "flag": False, "label": "a"}]
    with dspy.context(disable_history=False):
        first = predictor(text="public synthetic text")
        second = predictor(text="public synthetic text")
        third = await predictor.acall(text="another public synthetic text")
    assert len(backend) == 2
    assert (first.flag.value if rich else first.flag) is True
    assert (first.label.value if rich else first.label) == "b"
    assert first.rating.value == pytest.approx(1.5)
    assert first.rating.probabilities == {0: 0.1, 1: 0.3, 2: 0.6}
    assert first.rating.confidence == 0.61
    assert first.toDict() == second.toDict() == third.toDict()
    assert first.get_decision_evidence() == ANSWERS
    evidence = first.get_decision_evidence()
    evidence["flag"]["noul"] = 0
    assert first.get_decision_evidence()["flag"]["noul"] == 0.8
    assert backend[0]["state"]["inputs"] == {"text": "public synthetic text"}
    assert backend[0]["state"]["demos"] == predictor.demos
    assert "temperature" not in backend[0] and "messages" not in backend[0]
    assert lm.history[1].response.cache_hit
    assert lm.history[1].response.provider_data["systemone"]["answers"] == ANSWERS
    restored = dspy.LMRequest.model_validate_json(lm.history[0].request.model_dump_json())
    assert restored.decision.questions["flag"]["type"] == "noul"
    response = JsonCodec().loads(JsonCodec().dumps(lm.history[0].response))
    assert response.provider_data["systemone"]["answers"] == ANSWERS


def test_cache_survives_disk_and_isolates_endpoint_and_questions(backend, tmp_path, monkeypatch):
    lm = dspy.LM("typesafe/jev-test", model_type="decision", api_key="secret", api_base="https://one.example.test")
    predictor = dspy.Predict(signature())
    predictor(text="x", lm=lm)
    monkeypatch.setattr(
        dspy, "cache", Cache(disk_cache_dir=str(tmp_path), enable_disk_cache=True, enable_memory_cache=True)
    )
    predictor(text="x", lm=lm)
    assert len(backend) == 1
    predictor(text="x", lm=lm.copy(api_base="https://two.example.test"))
    assert len(backend) == 2
    predictor.fields["flag"] = {"instructions": "Different question."}
    predictor(text="x", lm=lm)
    assert len(backend) == 3


def test_parameters_roundtrip_without_credentials(backend, tmp_path):
    predictor = dspy.Predict(signature())
    predictor.set_lm(dspy.LM("typesafe/jev-test", model_type="decision", api_key="test-secret"))
    predictor.fields = {"flag": {"threshold": 0.9}, "label": {"weights": {"b": 0.1}}, "rating": {"cuts": [0.4, 1.6]}}
    first = predictor(text="x")
    assert first.flag.value is False and first.label.value == "a" and first.rating.level == 1
    predictor.save(tmp_path / "program.json")
    assert "test-secret" not in (tmp_path / "program.json").read_text()
    restored = dspy.Predict(signature())
    restored.load(tmp_path / "program.json")
    assert restored.lm.model_type == "decision"
    assert restored.fields == predictor.fields
    assert restored(text="x").toDict() == first.toDict()
    assert json.loads((tmp_path / "program.json").read_text())["fields"] == predictor.fields


@pytest.mark.parametrize(
    "config",
    [
        {"temperature": 0},
        {"n": 2},
        {"stream": True},
        {"tools": []},
        {"rollout_id": 1},
        {"max_tokens": 3},
        {"response_format": {}},
    ],
)
def test_generation_parameters_fail_before_transport(backend, config):
    lm = dspy.LM("typesafe/jev-test", model_type="decision")
    with pytest.raises((ValueError, dspy.LMUnsupportedFeatureError)):
        dspy.Predict(signature())(text="x", lm=lm, config=config)
    assert not backend


@pytest.mark.parametrize(
    "module", [lambda: dspy.Predict("text -> answer"), lambda: dspy.ChainOfThought("text -> flag: bool")]
)
def test_no_freeform_or_cot_fallback(backend, module):
    lm = dspy.LM("typesafe/jev-test", model_type="decision")
    with pytest.raises(ValueError, match="Unsupported System One output"):
        module()(text="x", lm=lm)
    assert not backend


def test_cache_is_partitioned_by_credentials_and_missing_auth(backend):
    predictor = dspy.Predict(signature())
    lm = dspy.LM("typesafe/jev-test", model_type="decision", api_key="key-one")
    predictor(text="x", lm=lm)
    predictor(text="x", lm=lm.copy(api_key="key-two"))
    predictor(text="x", lm=lm.copy(api_key=None))
    assert len(backend) == 3


@pytest.mark.parametrize("value", [object(), {1: "bad key"}, {"item": object()}, float("nan"), b"bytes"])
def test_unsupported_state_rejected_without_stringification(backend, value):
    lm = dspy.LM("typesafe/jev-test", model_type="decision", api_key="key-one")
    with pytest.raises(ValueError):
        dspy.Predict(signature())(text=value, lm=lm)
    assert not backend


def test_nested_pydantic_media_rejected_before_serializers_hide_it(backend):
    from pydantic import BaseModel

    class Container(BaseModel):
        image: dspy.Image

    value = Container(image=dspy.Image("https://example.test/image.png"))
    lm = dspy.LM("typesafe/jev-test", model_type="decision", api_key="key-one")
    with pytest.raises(ValueError, match="media"):
        dspy.Predict(signature())(text=value, lm=lm)
    assert not backend


def test_per_call_history_is_sanitized_and_runtime_state_is_not_saved(backend, tmp_path):
    lm = dspy.LM("typesafe/jev-test", model_type="decision", cache=False)
    predictor = dspy.Predict(signature())
    with dspy.context(disable_history=False):
        predictor(text="x", lm=lm, config={"api_key": "per-call-secret"})
    saved_history = lm.history[-1].model_dump_json()
    assert "per-call-secret" not in saved_history
    assert "state" in saved_history and "questions" in saved_history
    lm.kwargs.update(azure_ad_token_provider=lambda: "runtime-secret", http_client=object())
    predictor.set_lm(lm)
    predictor.save(tmp_path / "runtime.json")
    state = json.loads((tmp_path / "runtime.json").read_text())["lm"]
    assert "azure_ad_token_provider" not in state and "http_client" not in state


def test_decision_requires_native_request_and_preserves_generic_capabilities(backend):
    lm = dspy.LM("typesafe/jev-test", model_type="decision")
    assert not lm.supports_function_calling and not lm.supports_response_schema and not lm.supports_reasoning
    with pytest.raises((ValueError, dspy.LMUnsupportedFeatureError)):
        lm("free text")
    with pytest.raises(dspy.LMUnsupportedModelError):
        dspy.LM("openai/gpt-4o-mini", model_type="decision")
    assert not backend
