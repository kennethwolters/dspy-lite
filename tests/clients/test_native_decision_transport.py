"""Predict → BaseLM → LiteLM → HTTP contract without a live provider."""

import json

import httpx2
import pytest

import dspy
from dspy.utils.callback import BaseCallback


def predictor():
    return dspy.Predict(
        dspy.Signature(
            {
                "text": (str, dspy.InputField()),
                "flag": (bool, dspy.OutputField(desc="Is it actionable?")),
            },
            "Assess the text.",
        )
    )


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_full_native_http_boundary_and_callbacks(async_call):
    events = []

    class Callback(BaseCallback):
        def on_lm_start(self, call_id, instance, inputs):
            events.append(("start", call_id))

        def on_lm_end(self, call_id, outputs, exception=None):
            assert exception is None and isinstance(outputs, dspy.LMResponse)
            events.append(("end", call_id))

    def handle(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://resource.example.test/providers/microsoft/v1/systemone"
        assert request.headers["api-key"] == "test-key"
        assert body["model"] == "decision-deployment" and body["state"]["inputs"] == {"text": "synthetic text"}
        assert body["questions"]["flag"] == {"type": "noul", "instructions": "Is it actionable?"}
        assert "messages" not in body
        return httpx2.Response(
            200,
            json={"answers": {"flag": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 4, "output_tokens": 1}},
        )

    client_type = httpx2.AsyncClient if async_call else httpx2.Client
    client = client_type(transport=httpx2.MockTransport(handle))
    lm = dspy.LM(
        "azure_ai/decision-deployment",
        model_type="decision",
        api_key="test-key",
        api_base="https://resource.example.test",
        http_client=client,
        cache=False,
        num_retries=0,
        callbacks=[Callback()],
    )
    with dspy.context(disable_history=False):
        result = (
            await predictor().acall(text="synthetic text", lm=lm)
            if async_call
            else predictor()(text="synthetic text", lm=lm)
        )
    assert result.flag is True
    assert result.get_decision_evidence()["flag"]["noul"] == 0.9
    assert events[0][0] == "start" and events[1] == ("end", events[0][1])
    assert lm.history[-1].response.usage.input_tokens == 4
    assert not client.is_closed
    if async_call:
        await client.aclose()
    else:
        client.close()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "status,body,error_type",
    [
        (401, {"error": "private payload test-key"}, dspy.LMAuthError),
        (429, {"error": "private payload test-key"}, dspy.LMRateLimitError),
        (200, {"answers": {"flag": {"type": "noul", "noul": "private payload test-key"}}}, dspy.LMError),
        (200, {"answers": {"wrong": {"type": "noul", "noul": 0.8}}}, dspy.LMError),
    ],
)
@pytest.mark.asyncio
async def test_failure_sanitized_without_fallback(async_call, status, body, error_type):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx2.Response(status, json=body)

    client = (httpx2.AsyncClient if async_call else httpx2.Client)(transport=httpx2.MockTransport(handle))
    lm = dspy.LM(
        "typesafe/jev-test", model_type="decision", api_key="test-key", http_client=client, cache=False, num_retries=0
    )
    with pytest.raises(error_type) as caught:
        if async_call:
            await predictor().acall(text="private payload", lm=lm)
        else:
            predictor()(text="private payload", lm=lm)
    assert len(calls) == 1 and calls[0].endswith("/v1/systemone")
    assert "private payload" not in str(caught.value) and "test-key" not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    if async_call:
        await client.aclose()
    else:
        client.close()


def test_explicit_decision_request_is_never_chat():
    request = dspy.LMRequest(
        model="openai/gpt-4o-mini",
        messages=[],
        decision=dspy.LMDecisionSpec(
            state="synthetic", questions={"flag": {"type": "noul", "instructions": "Relevant?"}}
        ),
    )
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="model_type='decision'"):
        dspy.LM("openai/gpt-4o-mini")(request)
