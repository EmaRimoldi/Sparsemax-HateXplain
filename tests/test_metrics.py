import numpy as np
import pytest

from hatexplain_sparsemax.metrics import (
    compare_arrays,
    record_metrics,
    spans_to_mask,
    union_rationales,
)


def test_union_and_span_masks():
    np.testing.assert_array_equal(union_rationales([[0, 1, 0], [0, 0, 1]]), [0, 1, 1])
    np.testing.assert_array_equal(
        spans_to_mask([{"start_token": 1, "end_token": 3}], 4),
        [0, 1, 1, 0],
    )


def test_record_metrics_use_gold_class_and_union_rationale():
    dataset_record = {
        "annotators": [
            {"label": "offensive"},
            {"label": "offensive"},
            {"label": "normal"},
        ],
        "rationales": [[0, 1, 0], [0, 0, 1]],
    }
    explanation = {
        "classification_scores": {"normal": 0.2, "offensive": 0.7},
        "sufficiency_classification_scores": {"normal": 0.1, "offensive": 0.6},
        "comprehensiveness_classification_scores": {"normal": 0.3, "offensive": 0.4},
        "rationales": [
            {"hard_rationale_predictions": [{"start_token": 1, "end_token": 3}]}
        ],
    }

    metrics = record_metrics(explanation, dataset_record)
    assert metrics["sufficiency"] == pytest.approx(0.1)
    assert metrics["comprehensiveness"] == pytest.approx(0.3)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.0


def test_paired_statistics_report_softmax_minus_sparsemax():
    softmax = np.array([1.0, 2.0, 3.0])
    sparsemax = np.array([0.4, 1.5, 2.6])
    result = compare_arrays(softmax, sparsemax)

    assert result["mean_difference_softmax_minus_sparsemax"] == 0.5
    assert result["wilcoxon_pvalue"] <= 1.0
