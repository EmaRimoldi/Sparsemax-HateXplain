from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import precision_recall_fscore_support

from .data import majority_label

METRICS = ("sufficiency", "comprehensiveness", "precision", "recall", "f1")


def load_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            post_id = record["annotation_id"]
            if post_id in records:
                raise ValueError(f"duplicate annotation_id {post_id!r} at line {line_number}")
            records[post_id] = record
    return records


def union_rationales(rationales: Iterable[Iterable[int]]) -> np.ndarray:
    masks = [np.asarray(mask, dtype=np.int64) for mask in rationales]
    if not masks:
        return np.array([], dtype=np.int64)
    return np.logical_or.reduce(masks).astype(np.int64)


def spans_to_mask(spans: Iterable[dict[str, int]], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.int64)
    for span in spans:
        start = max(0, int(span["start_token"]))
        end = min(length, int(span["end_token"]))
        if end > start:
            mask[start:end] = 1
    return mask


def score_for_label(scores: Any, label: str) -> float:
    if isinstance(scores, dict):
        return float(scores[label])
    raise TypeError("classification scores must be a mapping keyed by class label")


def record_metrics(
    explanation: dict[str, Any],
    dataset_record: dict[str, Any],
) -> dict[str, float]:
    gold_label = majority_label(dataset_record["annotators"])
    if gold_label is None:
        raise ValueError("cannot score a record without a majority label")

    original = score_for_label(explanation["classification_scores"], gold_label)
    sufficient = score_for_label(
        explanation["sufficiency_classification_scores"],
        gold_label,
    )
    comprehensive = score_for_label(
        explanation["comprehensiveness_classification_scores"],
        gold_label,
    )

    gold_mask = union_rationales(dataset_record.get("rationales", []))
    spans = explanation["rationales"][0]["hard_rationale_predictions"]
    predicted_mask = spans_to_mask(spans, len(gold_mask))
    if gold_mask.size:
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_mask,
            predicted_mask,
            average="binary",
            zero_division=0,
        )
    else:
        precision = recall = f1 = 0.0

    return {
        "sufficiency": original - sufficient,
        "comprehensiveness": original - comprehensive,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def paired_metrics(
    dataset_path: str | Path,
    splits_path: str | Path,
    softmax_path: str | Path,
    sparsemax_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    with Path(dataset_path).open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    with Path(splits_path).open(encoding="utf-8") as handle:
        test_ids = set(json.load(handle)["test"])

    softmax = {key: value for key, value in load_jsonl(softmax_path).items() if key in test_ids}
    sparsemax = {key: value for key, value in load_jsonl(sparsemax_path).items() if key in test_ids}
    if softmax.keys() != sparsemax.keys():
        missing_sparse = sorted(softmax.keys() - sparsemax.keys())
        missing_soft = sorted(sparsemax.keys() - softmax.keys())
        raise ValueError(
            "paired ID mismatch: "
            f"{len(missing_sparse)} absent from sparsemax, "
            f"{len(missing_soft)} absent from softmax"
        )
    if not softmax:
        raise ValueError("no paired test explanations were found")

    ordered_ids = sorted(softmax)
    soft_rows = [record_metrics(softmax[post_id], dataset[post_id]) for post_id in ordered_ids]
    sparse_rows = [record_metrics(sparsemax[post_id], dataset[post_id]) for post_id in ordered_ids]
    soft_arrays = {
        metric: np.asarray([row[metric] for row in soft_rows], dtype=np.float64)
        for metric in METRICS
    }
    sparse_arrays = {
        metric: np.asarray([row[metric] for row in sparse_rows], dtype=np.float64)
        for metric in METRICS
    }
    return soft_arrays, sparse_arrays, ordered_ids


def compare_arrays(softmax: np.ndarray, sparsemax: np.ndarray) -> dict[str, float]:
    difference = softmax - sparsemax
    t_statistic, t_pvalue = ttest_rel(softmax, sparsemax)
    if np.allclose(difference, 0):
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        wilcoxon_statistic, wilcoxon_pvalue = wilcoxon(softmax, sparsemax)
    difference_std = difference.std(ddof=1)
    return {
        "softmax_mean": float(softmax.mean()),
        "softmax_std": float(softmax.std(ddof=1)),
        "sparsemax_mean": float(sparsemax.mean()),
        "sparsemax_std": float(sparsemax.std(ddof=1)),
        "mean_difference_softmax_minus_sparsemax": float(difference.mean()),
        "paired_t_statistic": float(t_statistic),
        "paired_t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(wilcoxon_statistic),
        "wilcoxon_pvalue": float(wilcoxon_pvalue),
        "cohen_dz": float(difference.mean() / difference_std)
        if difference_std > 0
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired LIME explanation outputs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--softmax", required=True)
    parser.add_argument("--sparsemax", required=True)
    args = parser.parse_args()

    softmax, sparsemax, post_ids = paired_metrics(
        args.dataset,
        args.splits,
        args.softmax,
        args.sparsemax,
    )
    result = {
        "paired_samples": len(post_ids),
        "bonferroni_alpha_for_five_metrics": 0.01,
        "metrics": {
            metric: compare_arrays(softmax[metric], sparsemax[metric]) for metric in METRICS
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
