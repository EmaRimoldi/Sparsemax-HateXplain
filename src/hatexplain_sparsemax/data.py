from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .sparsemax import sparsemax_numpy

LABELS = ("hatespeech", "normal", "offensive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def majority_label(annotations: Iterable[dict[str, Any]]) -> str | None:
    """Return a deterministic two-of-three label, or None when there is no majority."""
    counts = Counter(annotation["label"] for annotation in annotations)
    if not counts:
        return None
    label, count = counts.most_common(1)[0]
    return label if count >= 2 else None


def aggregate_word_rationale(
    rationales: list[list[int]],
    token_count: int,
    method: str,
) -> np.ndarray:
    """Aggregate three annotator masks while making missing annotations explicit."""
    if len(rationales) > 3:
        raise ValueError("HateXplain records must not contain more than three rationale masks")
    padded = [list(mask) for mask in rationales]
    padded.extend([[0] * token_count for _ in range(3 - len(padded))])
    masks = np.asarray(padded, dtype=np.float64)
    if masks.ndim != 2 or masks.shape[1] != token_count:
        raise ValueError("rationale masks must align with post_tokens")
    if method == "mean":
        return masks.mean(axis=0)
    if method == "majority":
        return (masks.sum(axis=0) >= 2).astype(np.float64)
    if method == "union":
        return (masks.sum(axis=0) >= 1).astype(np.float64)
    if method.startswith("annotator_"):
        index = int(method.rsplit("_", maxsplit=1)[1])
        if index not in {0, 1, 2}:
            raise ValueError(f"unsupported annotator index: {index}")
        return masks[index]
    raise ValueError(f"unsupported rationale aggregation: {method}")


def mean_word_rationale(rationales: list[list[int]], token_count: int) -> np.ndarray:
    """Backward-compatible alias for the historical mean aggregation."""
    return aggregate_word_rationale(rationales, token_count, "mean")


def length_matched_random_rationale(
    human_scores: np.ndarray,
    post_id: str,
    seed: int,
) -> np.ndarray:
    """Sample a deterministic random mask with the human union's token count."""
    scores = np.asarray(human_scores, dtype=np.float64)
    selected = np.flatnonzero(scores > 0)
    output = np.zeros_like(scores)
    if selected.size == 0:
        return output

    available = np.flatnonzero(scores <= 0)
    candidates = available if available.size >= selected.size else np.arange(scores.size)
    digest = hashlib.sha256(f"{seed}:{post_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], byteorder="big"))
    sampled = rng.choice(candidates, size=selected.size, replace=False)
    output[sampled] = 1.0
    return output


def normalize_rationale_target(
    token_scores: np.ndarray,
    attention_mask: np.ndarray,
    method: str,
    normal_post: bool,
) -> np.ndarray:
    """Create the effective token target, including active special tokens."""
    scores = np.asarray(token_scores, dtype=np.float64)
    active = np.asarray(attention_mask, dtype=bool)
    if scores.shape != active.shape:
        raise ValueError("token scores and attention mask must have the same shape")
    if not active.any():
        raise ValueError("at least one token must be active")

    target = np.zeros_like(scores, dtype=np.float64)
    if normal_post:
        target[active] = 1.0 / active.sum()
        return target

    active_scores = scores[active]
    if method == "softmax":
        shifted = active_scores - active_scores.max()
        values = np.exp(shifted)
        values /= values.sum()
    elif method == "sparsemax":
        values = sparsemax_numpy(active_scores)
    elif method == "binary":
        values = active_scores
    else:
        raise ValueError(f"unsupported target normalization: {method}")
    target[active] = values
    return target


class HateXplainDataset(Dataset[dict[str, Any]]):
    """Official HateXplain split with wordpiece-aligned rationale targets."""

    def __init__(
        self,
        dataset_path: str | Path,
        splits_path: str | Path,
        split: str,
        tokenizer: Any,
        max_length: int,
        target_normalization: str,
        rationale_source: str = "human",
        rationale_aggregation: str = "mean",
        random_rationale_seed: int = 1729,
        normal_rationale_policy: str = "ignore",
    ) -> None:
        with Path(dataset_path).open(encoding="utf-8") as handle:
            self.records: dict[str, dict[str, Any]] = json.load(handle)
        with Path(splits_path).open(encoding="utf-8") as handle:
            split_ids: dict[str, list[str]] = json.load(handle)

        if split not in split_ids:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(split_ids)}")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer is required for word-to-wordpiece alignment")

        self.ids = split_ids[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_normalization = target_normalization
        self.rationale_source = rationale_source
        self.rationale_aggregation = rationale_aggregation
        self.random_rationale_seed = random_rationale_seed
        self.normal_rationale_policy = normal_rationale_policy

        missing = [post_id for post_id in self.ids if post_id not in self.records]
        if missing:
            raise ValueError(f"{len(missing)} split IDs are absent from the dataset")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        post_id = self.ids[index]
        record = self.records[post_id]
        label = majority_label(record["annotators"])
        if label is None:
            raise ValueError(f"split record {post_id} has no majority label")

        words = record["post_tokens"] or ["dummy"]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        word_ids = encoding.word_ids()
        word_scores = aggregate_word_rationale(
            record.get("rationales", []),
            len(words),
            self.rationale_aggregation,
        )
        if self.rationale_source == "random":
            word_scores = length_matched_random_rationale(
                word_scores,
                post_id,
                self.random_rationale_seed,
            )
        elif self.rationale_source == "none":
            word_scores = np.zeros_like(word_scores)
        elif self.rationale_source != "human":
            raise ValueError(f"unsupported rationale source: {self.rationale_source}")
        aligned = np.zeros(self.max_length, dtype=np.float64)
        for token_index, word_index in enumerate(word_ids):
            if word_index is not None and word_index < len(word_scores):
                aligned[token_index] = word_scores[word_index]

        attention_mask = np.asarray(encoding["attention_mask"], dtype=np.int64)
        rationale_target = normalize_rationale_target(
            aligned,
            attention_mask,
            method=self.target_normalization,
            normal_post=label == "normal",
        )
        rationale_weight = float(
            self.rationale_source != "none"
            and not (label == "normal" and self.normal_rationale_policy == "ignore")
        )

        return {
            "post_id": post_id,
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "rationale_target": torch.tensor(rationale_target, dtype=torch.float32),
            "rationale_weight": torch.tensor(rationale_weight, dtype=torch.float32),
            "label": torch.tensor(LABEL_TO_ID[label], dtype=torch.long),
        }

    @property
    def label_ids(self) -> list[int]:
        labels = []
        for post_id in self.ids:
            label = majority_label(self.records[post_id]["annotators"])
            if label is None:
                raise ValueError(f"split record {post_id} has no majority label")
            labels.append(LABEL_TO_ID[label])
        return labels


def balanced_class_weights(label_ids: Iterable[int], num_classes: int = 3) -> torch.Tensor:
    labels = list(label_ids)
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("every class must occur in the training labels")
    weights = len(labels) / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def dataset_statistics(dataset_path: str | Path, splits_path: str | Path) -> dict[str, Any]:
    with Path(dataset_path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    with Path(splits_path).open(encoding="utf-8") as handle:
        splits = json.load(handle)

    stats: dict[str, Any] = {
        "dataset_records": len(records),
        "without_majority": sum(
            majority_label(record["annotators"]) is None for record in records.values()
        ),
        "splits": {},
    }
    for split_name, post_ids in splits.items():
        counts = Counter(majority_label(records[post_id]["annotators"]) for post_id in post_ids)
        if None in counts:
            raise ValueError(f"official split {split_name} contains records without a majority")
        stats["splits"][split_name] = {
            "total": len(post_ids),
            "labels": {label: counts[label] for label in LABELS},
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify HateXplain split and label counts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    args = parser.parse_args()
    print(json.dumps(dataset_statistics(args.dataset, args.splits), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
