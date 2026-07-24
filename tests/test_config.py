import json

import pytest

from hatexplain_sparsemax.config import load_config


def test_checked_in_configs_are_valid():
    softmax = load_config("configs/softmax.json")
    sparsemax = load_config("configs/sparsemax.json")

    assert softmax.epochs == sparsemax.epochs == 10
    assert softmax.supervised_heads == sparsemax.supervised_heads == 6
    assert softmax.seed == sparsemax.seed == 42
    assert softmax.normal_rationale_policy == sparsemax.normal_rationale_policy == "uniform"


def test_invalid_config_fails_fast(tmp_path):
    payload = json.loads(open("configs/sparsemax.json", encoding="utf-8").read())
    payload["supervised_heads"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="head"):
        load_config(path)


def test_no_rationale_requires_zero_auxiliary_weight(tmp_path):
    payload = json.loads(open("configs/sparsemax.json", encoding="utf-8").read())
    payload["rationale_source"] = "none"
    path = tmp_path / "invalid-none.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="attention_lambda=0"):
        load_config(path)
