"""Typed decision LM lifecycle; all HTTP/auth/retry work belongs to LiteLM."""

import os
from hashlib import sha256

import dspy
from dspy.clients._litelm import get_litelm
from dspy.core.types import LMOutput, LMRequest, LMResponse
from dspy.dsp.utils.settings import settings
from dspy.utils.exceptions import LMUnsupportedFeatureError

_TRANSPORT = {"api_key", "api_base", "azure_ad_token_provider", "timeout", "http_client"}


def _unsupported(lm):
    return LMUnsupportedFeatureError(
        "Native decisions require a decision request and do not support generation settings.", model=lm.model
    )


def prepare(lm, request):
    if (
        not isinstance(request, LMRequest)
        or request.decision is None
        or request.messages
        or request.tools
        or request.model != lm.model
        or settings.send_stream is not None
    ):
        raise _unsupported(lm)
    config = request.config.model_dump(exclude_none=True)
    extensions = config.pop("extensions", {})
    cache_config = config.pop("cache", None)
    if config or (cache_config and cache_config.get("rollout_id") is not None):
        raise _unsupported(lm)
    defaults = {k: v for k, v in lm.kwargs.items() if not (k in {"temperature", "max_tokens"} and v is None)}
    params = {**defaults, **extensions}
    if params.keys() - _TRANSPORT:
        raise _unsupported(lm)
    provider = lm.model.split("/", 1)[0]
    params["api_base"] = (
        params.get("api_base")
        or os.getenv(f"{provider.upper()}_API_BASE")
        or ("https://api.typesafe.ai" if provider == "typesafe" else None)
    )
    payload = {"model": lm.model, **request.decision.model_dump(), **params, "num_retries": lm.num_retries}
    # Client and auth objects never enter cache hashing/deepcopy or persisted payloads.
    key = {k: v for k, v in payload.items() if k not in {"api_key", "azure_ad_token_provider", "http_client"}}
    key["_fn_identifier"] = "dspy.clients.decision.systemone.v1"
    enabled = lm.cache if not cache_config or cache_config.get("enabled") is None else cache_config["enabled"]
    credential = params.get("api_key")
    if credential is None:
        credential = os.getenv(f"{provider.upper()}_API_KEY")
    if (
        not isinstance(credential, str)
        or not credential
        or params.get("azure_ad_token_provider") is not None
        or params.get("http_client") is not None
    ):
        # Callback/client authentication identity is not stable or inspectable.
        enabled = False
    else:
        # Only a one-way credential namespace enters the already hashed cache key.
        key["_auth_namespace"] = sha256(credential.encode()).hexdigest()
    return payload, key, enabled


def _validate(litelm, response, payload):
    spec = litelm.SystemOneRequest(
        model=payload["model"].split("/", 1)[1], state=payload["state"], questions=payload["questions"]
    )
    return response.validate_for(spec)


def _normalize(litelm, response, payload):
    response = _validate(litelm, response, payload)
    return LMResponse(
        model=response.model or payload["model"],
        outputs=[LMOutput()],
        usage=response.usage.model_dump(exclude_none=True) if response.usage is not None else None,
        cache_hit=response.cache_hit,
        provider_data={"systemone": response.model_dump(mode="json")},
    )


def _cached(litelm, key, payload, enabled):
    response = dspy.cache.get(key, ["api_key"]) if enabled else None
    if response is not None:
        _validate(litelm, litelm.SystemOneResponse.model_validate(response.provider_data["systemone"]), payload)
    return response


def _safe_error(lm, exc):
    wrapped = lm._wrap_litelm_exception(exc)
    return type(wrapped)(
        message="Native decision inference failed.",
        model=lm.model,
        provider=lm.model.split("/", 1)[0],
        status=wrapped.status,
        retry_after=wrapped.retry_after,
    )


def _raise_clean(error):
    try:
        raise error from None
    finally:
        error.__context__ = None


def forward(lm, request):
    payload, key, enabled = prepare(lm, request)
    litelm = get_litelm(feature="native decision inference")
    error = None
    try:
        result = _cached(litelm, key, payload, enabled)
        if result is not None:
            return result
        result = _normalize(litelm, litelm.systemone(**payload), payload)
    except Exception as exc:
        error = _safe_error(lm, exc)
    if error is not None:
        _raise_clean(error)
    if enabled:
        dspy.cache.put(key, result, ["api_key"])
    return result


async def aforward(lm, request):
    payload, key, enabled = prepare(lm, request)
    litelm = get_litelm(feature="native decision inference")
    error = None
    try:
        result = _cached(litelm, key, payload, enabled)
        if result is not None:
            return result
        result = _normalize(litelm, await litelm.asystemone(**payload), payload)
    except Exception as exc:
        error = _safe_error(lm, exc)
    if error is not None:
        _raise_clean(error)
    if enabled:
        dspy.cache.put(key, result, ["api_key"])
    return result
