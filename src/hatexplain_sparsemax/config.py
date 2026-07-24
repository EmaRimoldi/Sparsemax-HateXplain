from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TargetNormalization = Literal["softmax", "sparsemax"]
AttentionSource = Literal["post_softmax", "raw_scores"]
AttentionLoss = Literal["cross_entropy", "sparsemax"]


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

    def validate(self) -> None:
        if not self.variant:
            raise ValueError("variant must not be empty")
        if self.target_normalization not in {"softmax", "sparsemax"}:
            raise ValueError(f"unsupported target normalization: {self.target_normalization}")
        if self.attention_source not in {"post_softmax", "raw_scores"}:
            raise ValueError(f"unsupported attention source: {self.attention_source}")
        if self.attention_loss not in {"cross_entropy", "sparsemax"}:
            raise ValueError(f"unsupported attention loss: {self.attention_loss}")
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    config = ExperimentConfig(**payload)
    config.validate()
    return config
