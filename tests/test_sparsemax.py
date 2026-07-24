import numpy as np
import torch

from hatexplain_sparsemax.sparsemax import (
    soft_target_cross_entropy,
    sparsemax,
    sparsemax_loss,
    sparsemax_numpy,
)


def test_numpy_sparsemax_has_exact_zeros_and_unit_mass():
    result = sparsemax_numpy(np.array([0.0, 1 / 3, 1.0, 2 / 3, 0.0]))
    np.testing.assert_allclose(result, [0, 0, 2 / 3, 1 / 3, 0], atol=1e-7)
    assert result.sum() == 1.0


def test_torch_sparsemax_supports_masks_and_gradients():
    scores = torch.tensor([[1.0, 0.0, 100.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    result = sparsemax(scores, mask=mask)

    assert torch.allclose(result, torch.tensor([[1.0, 0.0, 0.0]]))
    result.sum().backward()
    assert scores.grad is not None


def test_sparsemax_loss_is_zero_at_matching_vertex():
    scores = torch.tensor([[1.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    loss = sparsemax_loss(scores, target)

    assert torch.allclose(loss, torch.tensor(0.0))
    loss.backward()
    assert scores.grad is not None


def test_historical_soft_target_loss_masks_padding():
    values = torch.tensor([[0.8, 0.2, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    expected = -torch.log_softmax(values[0, :2], dim=0)[0]

    assert torch.allclose(soft_target_cross_entropy(values, target, mask), expected)
