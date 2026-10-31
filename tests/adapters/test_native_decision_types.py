"""Decision values retain typed answer spaces and unmodified probability evidence."""

from typing import Annotated, Literal

import pytest
from pydantic import TypeAdapter, ValidationError

import dspy
from dspy.adapters.decision_state import DecisionState
from dspy.adapters.types.decision import Choice, Noul, Score


@pytest.mark.parametrize("selected", [1, True, None, "雪"])
def test_choice_typed_values_and_metadata(selected):
    kind = Choice[(1, "integer"), (True, "boolean"), (None, "missing"), ("雪", "snow")]
    value = kind(value=selected, confidence=0.7)
    assert type(value.value) is type(selected)
    assert kind.model_validate_json(value.model_dump_json()) == value
    native = TypeAdapter(Annotated[Literal[1, True, None, "雪"], kind]).validate_python(selected)
    assert type(native) is type(selected)


@pytest.mark.parametrize(
    "kind,options",
    [
        (Choice, ((1, "int"), ("1", "str"))),
        (Choice, (([], "bad"),)),
        (Noul, ((1, "bad"),)),
        (Noul, ((True, "a"), (True, "b"))),
        (Score, ("one",)),
        (Score, (None, "high")),
        (Score, (1, "high")),
        (Choice, (("a", {"bad": float("nan")}), ("b", "ok"))),
    ],
)
def test_invalid_declarations(kind, options):
    with pytest.raises(ValueError):
        kind[options]


def test_bare_bool_criteria_accessor_and_atomic_replacement():
    predictor = dspy.Predict(dspy.Signature({"flag": (bool, dspy.OutputField(desc="Relevant?"))}))
    assert predictor.get_criteria("flag") is None
    predictor.set_criteria("flag", {"true": {"examples": ["yes"]}})
    snapshot = predictor.get_criteria("flag")
    snapshot["true"]["examples"].clear()
    assert predictor.get_criteria("flag") == {"true": {"examples": ["yes"]}}
    with pytest.raises(ValueError):
        predictor.set_criteria("flag", {"wrong": "invalid"})
    assert predictor.get_criteria("flag") == {"true": {"examples": ["yes"]}}


def test_choice_cache_keeps_bool_and_int_separate():
    boolean = Choice[(True, "yes"), (False, "no")]
    integer = Choice[(1, "yes"), (0, "no")]
    assert boolean is not integer
    assert type(boolean(value=True, confidence=1).value) is bool
    assert type(integer(value=1, confidence=1).value) is int


def test_descriptions_snapshot_and_native_annotation_checks():
    description = {"examples": ["original"]}
    kind = Noul[(True, description)]
    description["examples"].append("mutated")
    assert kind.criteria() == {"true": {"examples": ["original"]}}
    returned = kind.criteria()
    returned["true"]["examples"].clear()
    assert kind.criteria()["true"]["examples"] == ["original"]
    with pytest.raises(ValueError, match="match the Literal"):
        TypeAdapter(Annotated[Literal["a", "c"], Choice[("a", "first"), ("b", "second")]])
    with pytest.raises(ValueError, match="Use Score"):
        TypeAdapter(Annotated[float, Score["low", "high"]])


@pytest.mark.parametrize(
    "field,config",
    [
        ("flag", {"threshold": -0.1}),
        ("flag", {"threshold": float("nan")}),
        ("flag", {"threshold": True}),
        ("flag", {"unknown": 1}),
        ("label", {"weights": {"missing": 1}}),
        ("label", {"weights": {"a": 0, "b": 0}}),
        ("label", {"weights": {"a": float("inf")}}),
        ("rating", {"cuts": [1.5, 0.5]}),
        ("rating", {"cuts": [0.5]}),
        ("rating", {"criteria": ["low", "high"]}),
        ("flag", {"criteria": {"yes": "wrong"}}),
        ("missing", {}),
    ],
)
def test_invalid_parameters_on_save_load_and_call(field, config):
    sig = dspy.Signature(
        {
            "flag": (Noul, dspy.OutputField(desc="Is it relevant?")),
            "label": (Choice[("a", "first"), ("b", "second")], dspy.OutputField(desc="Choose.")),
            "rating": (Score["low", "middle", "high"], dspy.OutputField(desc="Rate.")),
        }
    )
    predictor = dspy.Predict(sig)
    state = predictor.dump_state()
    state["fields"][field] = config
    with pytest.raises(ValueError):
        predictor.load_state(state)
    assert predictor.fields == {}
    predictor.fields[field] = config
    with pytest.raises(ValueError):
        predictor.dump_state()
    with pytest.raises(ValueError):
        predictor(lm=dspy.LM("typesafe/jev-test", model_type="decision"))


def test_decode_snapshots_threshold_weights_cuts_and_evidence():
    sig = dspy.Signature(
        {
            "flag": (Noul, dspy.OutputField(desc="Is it relevant?")),
            "label": (Choice[("a", "first"), ("b", "second")], dspy.OutputField(desc="Choose.")),
            "rating": (Score["low", "middle", "high"], dspy.OutputField(desc="Rate.")),
        }
    )
    fields = {"flag": {"threshold": 0.9}, "label": {"weights": {"a": 0.1}}, "rating": {"cuts": [0.4, 1.6]}}
    state = DecisionState(sig, fields, system_one=True)
    fields["flag"]["threshold"] = 0.1
    evidence = {
        "flag": {"noul": 0.8},
        "label": {"probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.73},
        "rating": {"probabilities": {0: 0.1, 1: 0.3, 2: 0.6}, "confidence": 0.61},
    }
    result = state._decode(evidence)
    assert not result["flag"]
    assert result["label"].value == "b" and result["label"].probabilities == {"a": 0.8, "b": 0.2}
    assert result["rating"].value == pytest.approx(1.5) and result["rating"].level == 1
    assert result["rating"].confidence == 0.61


@pytest.mark.parametrize("value", [-0.1, 2.1, float("nan")])
def test_score_bounds_and_roundtrip(value):
    rating = Score["low", "medium", "high"]
    with pytest.raises(ValidationError):
        rating(value=value, confidence=0.7)
    valid = rating(value=1.5, confidence=0.7, probabilities={0: 0.1, 1: 0.3, 2: 0.6}, level=2)
    assert rating.model_validate_json(valid.model_dump_json()) == valid
