import pytest
import torch
from transformers import BertConfig, BertModel

from hatexplain_sparsemax.config import ExperimentConfig
from hatexplain_sparsemax.model import SupervisedAttentionBert


def experiment(variant):
    sparse = variant == "sparsemax"
    return ExperimentConfig(
        variant=variant,
        model_name="tiny-test-bert",
        target_normalization="sparsemax" if sparse else "softmax",
        attention_source="raw_scores" if sparse else "post_softmax",
        attention_loss="sparsemax" if sparse else "cross_entropy",
        max_length=8,
        batch_size=2,
        epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        attention_lambda=0.001,
        supervised_layer=1,
        supervised_heads=2,
        seed=42,
        dataset_path="unused",
        splits_path="unused",
        output_dir="unused",
    )


@pytest.mark.parametrize("variant", ["softmax", "sparsemax"])
def test_tiny_model_forward_and_backward(variant):
    config = BertConfig(
        vocab_size=100,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = SupervisedAttentionBert(
        BertModel(config),
        experiment(variant),
        torch.ones(3),
    )
    input_ids = torch.randint(0, 100, (2, 8))
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )
    rationale_target = attention_mask.float()
    rationale_target /= rationale_target.sum(dim=1, keepdim=True)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([0, 2]),
        rationale_target=rationale_target,
    )

    assert output.logits.shape == (2, 3)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.rationale_loss is not None and torch.isfinite(output.rationale_loss)
    output.loss.backward()
    assert model.classifier.weight.grad is not None


def test_mse_can_supervise_an_explicit_head_and_ignore_examples():
    experiment_config = experiment("softmax")
    payload = experiment_config.to_dict()
    payload.update(
        attention_loss="mse",
        target_normalization="binary",
        supervised_heads=1,
        supervised_head_indices=(2,),
    )
    config = BertConfig(
        vocab_size=100,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = SupervisedAttentionBert(
        BertModel(config),
        ExperimentConfig(**payload),
        torch.ones(3),
    )
    attention_mask = torch.ones((2, 8), dtype=torch.bool)
    output = model(
        input_ids=torch.randint(0, 100, (2, 8)),
        attention_mask=attention_mask,
        labels=torch.tensor([0, 1]),
        rationale_target=torch.zeros((2, 8)),
        rationale_weight=torch.tensor([1.0, 0.0]),
    )

    assert output.rationale_loss is not None and torch.isfinite(output.rationale_loss)
