from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TargetNormalization = Literal["softmax", "sparsemax", "binary"]
AttentionSource = Literal["post_softmax", "raw_scores"]
AttentionLoss = Literal["cross_entropy", "sparsemax", "mse"]
RationaleSource = Literal["human", "random", "none"]
RationaleAggregation = Literal[
    "mean",
    "majority",
    "union",
    "annotator_0",
    "annotator_1",
    "annotator_2",
]
NormalRationalePolicy = Literal["uniform", "ignore"]


@dataclass(frozen=True)
class ExperimentConfig:
    variant: str
    model_name: str
    target_normalization: TargetNormalization
    attention_source: AttentionSource
    attention_loss: AttentionLoss
    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    dropout: float
    attention_lambda: float
    supervised_layer: int
    supervised_heads: int
    seed: int
    dataset_path: str
    splits_path: str
    output_dir: str
    rationale_source: RationaleSource = "human"
    rationale_aggregation: RationaleAggregation = "mean"
    random_rationale_seed: int = 1729
    normal_rationale_policy: NormalRationalePolicy = "ignore"
    supervised_head_indices: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not self.variant:
            raise ValueError("variant must not be empty")
        if self.target_normalization not in {"softmax", "sparsemax", "binary"}:
            raise ValueError(f"unsupported target normalization: {self.target_normalization}")
        if self.attention_source not in {"post_softmax", "raw_scores"}:
            raise ValueError(f"unsupported attention source: {self.attention_source}")
        if self.attention_loss not in {"cross_entropy", "sparsemax", "mse"}:
            raise ValueError(f"unsupported attention loss: {self.attention_loss}")
        if self.rationale_source not in {"human", "random", "none"}:
            raise ValueError(f"unsupported rationale source: {self.rationale_source}")
        if self.rationale_aggregation not in {
            "mean",
            "majority",
            "union",
            "annotator_0",
            "annotator_1",
            "annotator_2",
        }:
            raise ValueError(f"unsupported rationale aggregation: {self.rationale_aggregation}")
        if self.normal_rationale_policy not in {"uniform", "ignore"}:
            raise ValueError(
                f"unsupported normal-rationale policy: {self.normal_rationale_policy}"
            )
        if self.max_length < 3:
            raise ValueError("max_length must leave room for content and special tokens")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.attention_lambda < 0:
            raise ValueError("learning_rate must be positive and attention_lambda non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.supervised_layer < 0 or self.supervised_heads < 1:
            raise ValueError("supervised layer/head values must be non-negative")
        if self.random_rationale_seed < 0:
            raise ValueError("random_rationale_seed must be non-negative")
        if self.rationale_source == "none" and self.attention_lambda != 0:
            raise ValueError("rationale_source='none' requires attention_lambda=0")
        if self.supervised_head_indices is not None:
            head_indices = tuple(self.supervised_head_indices)
            if not head_indices or any(index < 0 for index in head_indices):
                raise ValueError("supervised_head_indices must contain non-negative indices")
            if len(head_indices) != len(set(head_indices)):
                raise ValueError("supervised_head_indices must be unique")
            if len(head_indices) != self.supervised_heads:
                raise ValueError(
                    "supervised_heads must equal the number of supervised_head_indices"
                )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    config = ExperimentConfig(**payload)
    config.validate()
    return config
