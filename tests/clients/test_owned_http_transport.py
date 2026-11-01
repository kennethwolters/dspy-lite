"""Exercise borrowed transports through DSPy, LiteLM and the real OpenAI SDK."""
import json

import httpx2 as httpx
import pytest

import dspy


class ToyAnswer(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def response(request, status=200):
    if status != 200:
        return httpx.Response(status, json={"error": {"message": "unavailable"}}, request=request)
    return httpx.Response(200, request=request, json={
        "id": "toy-completion", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"answer":"Vier"}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


def make_lm(provider, client):
    kwargs = {"api_base": "https://toy.invalid/", "api_key": "toy-key", "http_client": client}
    if provider == "azure":
        kwargs["api_version"] = "2025-01-01-preview"
    return dspy.LM(f"{provider}/gpt-4o-mini", cache=False, num_retries=0, callbacks=[], **kwargs)


def predictor(lm):
    predict = dspy.Predict(ToyAnswer, callbacks=[])
    predict.set_lm(lm)
    return predict


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_sync_json_predict_borrows_client(provider):
    seen = []
    def handle(request):
        seen.append(request)
        return response(request)
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as client:
        lm = make_lm(provider, client)
        copied = lm.copy()
        assert copied.kwargs["http_client"] is client
        with dspy.context(adapter=dspy.JSONAdapter(callbacks=[], use_native_function_calling=False),
                          disable_history=True, callbacks=[]):
            result = predictor(copied)(question="Was ist zwei plus zwei?")
        assert result.answer == "Vier"
        assert len(seen) == 1
        body = json.loads(seen[0].content)
        assert body["response_format"]["type"] == "json_schema"
        assert "http_client" not in body
        assert not client.is_closed
        assert not lm.history and not copied.history
        assert client.get("https://toy.invalid/reuse").status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "azure"])
async def test_async_json_predict_borrows_client(provider):
    seen = []
    async def handle(request):
        seen.append(request)
        return response(request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        lm = make_lm(provider, client)
        assert lm.copy().kwargs["http_client"] is client
        with dspy.context(adapter=dspy.JSONAdapter(callbacks=[], use_native_function_calling=False),
                          disable_history=True, callbacks=[], track_usage=True):
            result = await predictor(lm).acall(question="Was ist zwei plus zwei?")
        assert result.answer == "Vier"
        assert result.get_lm_usage()[lm.model]["total_tokens"] == 15
        assert len(seen) == 1
        assert json.loads(seen[0].content)["response_format"]["type"] == "json_schema"
        assert not client.is_closed and not lm.history
        assert (await client.get("https://toy.invalid/reuse")).status_code == 200


@pytest.mark.parametrize("status", [400, 429, 503])
@pytest.mark.asyncio
async def test_async_failure_does_not_retry_or_close_client(status):
    seen = []
    async def handle(request):
        seen.append(request)
        return response(request, status)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        lm = make_lm("azure", client)
        with dspy.context(disable_history=True, callbacks=[]):
            with pytest.raises(dspy.LMError):
                await lm.acall("Toy question")
        assert len(seen) == 1
        assert not client.is_closed and not lm.history


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_borrowed_transport_is_not_serialized(provider):
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        lm = make_lm(provider, client)
        state = lm.dump_state()
        assert "http_client" not in state
        assert "api_key" not in state
        json.dumps(state)
        restored = dspy.LM.load_state(state)
        assert "http_client" not in restored.kwargs
