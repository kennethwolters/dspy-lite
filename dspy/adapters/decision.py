"""Native decision translation, resolved before chat capability planning."""

import copy
import math

from pydantic import BaseModel

from dspy.adapters.decision_state import DecisionState
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.decision import Score, _Decision
from dspy.adapters.utils import get_field_description_string
from dspy.core.types import LMConfig, LMDecisionSpec, LMRequest
from dspy.dsp.utils.settings import settings


class DecisionCompletions(list):
    def __init__(self, values, evidence):
        super().__init__([values])
        self.evidence = copy.deepcopy(evidence)


def _json_state(value):
    if isinstance(value, Type) and not isinstance(value, _Decision):
        raise ValueError("Native decisions support JSON/text state, not media or tool objects.")
    if isinstance(value, _Decision):
        return _json_state(value.model_dump(mode="json"))
    if isinstance(value, BaseModel):
        # Inspect actual fields before custom serializers can hide nested media.
        return _json_state(dict(value))
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: _json_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_state(item) for item in value]
    raise ValueError("Native decision state requires finite JSON values and string object keys.")


class DecisionAdapter:
    """Map a decision-only signature to typed System One questions, with no fallback."""

    def __init__(self, signature, fields, declared_signature=None):
        self.state = DecisionState(signature, fields, system_one=True, declared_signature=declared_signature)

    def _prepare(self, lm, lm_kwargs, signature, demos, inputs):
        if settings.send_stream is not None:
            raise ValueError("Native decision streaming is unsupported.")
        questions = {
            name: self.state._question(name, signature.output_fields[name], kind)
            for name, kind in self.state.types.items()
        }
        state = {
            "instructions": signature.instructions,
            "input_fields": get_field_description_string(signature.input_fields),
            "inputs": {k: v for k, v in inputs.items() if k in signature.input_fields},
        }
        if demos:
            state["demos"] = [{k: v for k, v in demo.items() if k in signature.fields} for demo in demos]
        return LMRequest(
            model=lm.model,
            messages=[],
            decision=LMDecisionSpec(state=_json_state(state), questions=questions),
            config=LMConfig.from_kwargs(**lm_kwargs),
        )

    def _decode(self, response):
        evidence = copy.deepcopy(response.provider_data["systemone"]["answers"])
        answers = copy.deepcopy(evidence)
        for name, kind in self.state.types.items():
            if issubclass(kind, Score):
                answers[name]["probabilities"] = {int(k): v for k, v in answers[name]["probabilities"].items()}
        return DecisionCompletions(self.state._decode(answers), evidence)

    def __call__(self, lm, **kwargs):
        return self._decode(lm(self._prepare(lm, **kwargs)))

    async def acall(self, lm, **kwargs):
        return self._decode(await lm.acall(self._prepare(lm, **kwargs)))
