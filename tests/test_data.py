import numpy as np
import torch

from hatexplain_sparsemax.data import (
    balanced_class_weights,
    dataset_statistics,
    majority_label,
    mean_word_rationale,
    normalize_rationale_target,
)


def annotations(*labels):
    return [{"label": label} for label in labels]


def test_majority_vote_is_explicit_and_deterministic():
    assert majority_label(annotations("normal", "normal", "offensive")) == "normal"
    assert majority_label(annotations("normal", "offensive", "hatespeech")) is None


def test_mean_word_rationale_pads_an_absent_third_mask_with_zeros():
    rationales = [[0, 1, 1], [0, 0, 1]]
    np.testing.assert_allclose(mean_word_rationale(rationales, 3), [0, 1 / 3, 2 / 3])


def test_target_normalization_matches_paper_example():
    scores = np.array([0, 1 / 3, 1, 2 / 3, 0], dtype=float)
    mask = np.ones(5, dtype=int)

    soft = normalize_rationale_target(scores, mask, "softmax", normal_post=False)
    sparse = normalize_rationale_target(scores, mask, "sparsemax", normal_post=False)

    np.testing.assert_allclose(soft, [0.124, 0.173, 0.337, 0.242, 0.124], atol=5e-4)
    np.testing.assert_allclose(sparse, [0, 0, 2 / 3, 1 / 3, 0], atol=1e-7)


def test_normal_target_is_uniform_over_active_sequence():
    target = normalize_rationale_target(
        np.zeros(5),
        np.array([1, 1, 1, 0, 0]),
        "sparsemax",
        normal_post=True,
    )
    np.testing.assert_allclose(target, [1 / 3, 1 / 3, 1 / 3, 0, 0])


def test_class_weights_are_balanced_from_passed_labels():
    weights = balanced_class_weights([0, 0, 0, 1, 1, 2])
    assert torch.allclose(weights, torch.tensor([2 / 3, 1.0, 2.0]))


def test_official_dataset_statistics():
    stats = dataset_statistics("data/dataset.json", "data/post_id_divisions.json")
    assert stats["dataset_records"] == 20_148
    assert stats["without_majority"] == 919
    assert stats["splits"]["train"]["total"] == 15_383
    assert stats["splits"]["val"]["total"] == 1_922
    assert stats["splits"]["test"]["total"] == 1_924
