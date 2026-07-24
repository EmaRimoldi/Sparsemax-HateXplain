from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import BertConfig, BertModel

from .config import ExperimentConfig
from .sparsemax import soft_target_cross_entropy, sparsemax_loss


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    classification_loss: Tensor | None = None
    rationale_loss: Tensor | None = None


class SupervisedAttentionBert(nn.Module):
    """BERT classifier with an explicit auxiliary rationale objective."""

    def __init__(
        self,
        bert: BertModel,
        experiment: ExperimentConfig,
        class_weights: Tensor,
    ) -> None:
        super().__init__()
        self.bert = bert
        self.experiment = experiment
        if (
            experiment.attention_source == "post_softmax"
            and hasattr(self.bert, "set_attn_implementation")
        ):
            self.bert.set_attn_implementation("eager")
        self.dropout = nn.Dropout(experiment.dropout)
        self.classifier = nn.Linear(bert.config.hidden_size, 3)
        self.classifier.weight.data.normal_(mean=0.0, std=bert.config.initializer_range)
        self.classifier.bias.data.zero_()
        self.register_buffer("class_weights", class_weights.clone().detach())
        self._validate_architecture(bert.config)

    @classmethod
    def from_pretrained(
        cls,
        experiment: ExperimentConfig,
        class_weights: Tensor,
    ) -> SupervisedAttentionBert:
        bert = BertModel.from_pretrained(experiment.model_name)
        return cls(bert, experiment, class_weights)

    def _validate_architecture(self, config: BertConfig) -> None:
        if self.experiment.supervised_layer >= config.num_hidden_layers:
            raise ValueError(
                f"layer {self.experiment.supervised_layer} is unavailable in a "
                f"{config.num_hidden_layers}-layer BERT"
            )
        if self.experiment.supervised_heads > config.num_attention_heads:
            raise ValueError(
                f"requested {self.experiment.supervised_heads} heads, but BERT has "
                f"{config.num_attention_heads}"
            )
        if self.experiment.supervised_head_indices is not None and max(
            self.experiment.supervised_head_indices
        ) >= config.num_attention_heads:
            raise ValueError("a supervised head index is unavailable in this BERT")

    @property
    def _head_indices(self) -> tuple[int, ...]:
        if self.experiment.supervised_head_indices is not None:
            return tuple(self.experiment.supervised_head_indices)
        return tuple(range(self.experiment.supervised_heads))

    def _raw_cls_scores(self, hidden_states: tuple[Tensor, ...], attention_mask: Tensor) -> Tensor:
        layer_index = self.experiment.supervised_layer
        layer_input = hidden_states[layer_index]
        attention = self.bert.encoder.layer[layer_index].attention.self
        num_heads = self.bert.config.num_attention_heads
        head_size = self.bert.config.hidden_size // num_heads

        def split_heads(projected: Tensor) -> Tensor:
            batch_size, sequence_length, _ = projected.shape
            return projected.view(batch_size, sequence_length, num_heads, head_size).transpose(1, 2)

        query = split_heads(attention.query(layer_input))
        key = split_heads(attention.key(layer_input))
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(head_size)
        scores = scores[:, self._head_indices, 0, :]
        floor = torch.finfo(scores.dtype).min / 2
        return scores.masked_fill(~attention_mask[:, None, :].bool(), floor)

    def _supervised_values(
        self,
        attentions: tuple[Tensor, ...],
        hidden_states: tuple[Tensor, ...],
        attention_mask: Tensor,
    ) -> Tensor:
        if self.experiment.attention_source == "raw_scores":
            return self._raw_cls_scores(hidden_states, attention_mask)
        layer_attention = attentions[self.experiment.supervised_layer]
        return layer_attention[:, self._head_indices, 0, :]

    def _rationale_loss(
        self,
        values: Tensor,
        target: Tensor,
        attention_mask: Tensor,
        rationale_weight: Tensor,
    ) -> Tensor:
        losses = []
        for head in range(self.experiment.supervised_heads):
            head_values = values[:, head, :]
            if self.experiment.attention_loss == "sparsemax":
                loss = sparsemax_loss(
                    head_values,
                    target,
                    attention_mask,
                    reduction="none",
                )
            elif self.experiment.attention_loss == "cross_entropy":
                loss = soft_target_cross_entropy(
                    head_values,
                    target,
                    attention_mask,
                    reduction="none",
                )
            else:
                squared_error = (head_values - target).square()
                active = attention_mask.to(squared_error.dtype)
                loss = (squared_error * active).sum(dim=-1) / active.sum(dim=-1)
            losses.append(loss)
        per_example = torch.stack(losses).sum(dim=0)
        weights = rationale_weight.to(per_example.dtype)
        if not torch.any(weights > 0):
            return values.sum() * 0
        return (per_example * weights).sum() / weights.sum()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        rationale_target: Tensor | None = None,
        rationale_weight: Tensor | None = None,
    ) -> ModelOutput:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.pooler_output is None:
            raise RuntimeError("BERT pooler output is required for classification")
        logits = self.classifier(self.dropout(outputs.pooler_output))

        if labels is None:
            return ModelOutput(logits=logits)

        classification_loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
        )
        rationale_loss = None
        total_loss = classification_loss
        if rationale_target is not None and self.experiment.attention_lambda > 0:
            values = self._supervised_values(
                outputs.attentions,
                outputs.hidden_states,
                attention_mask,
            )
            if rationale_weight is None:
                rationale_weight = torch.ones(
                    rationale_target.shape[0],
                    device=rationale_target.device,
                )
            rationale_loss = self._rationale_loss(
                values,
                rationale_target,
                attention_mask,
                rationale_weight,
            )
            total_loss = total_loss + self.experiment.attention_lambda * rationale_loss

        return ModelOutput(
            logits=logits,
            loss=total_loss,
            classification_loss=classification_loss,
            rationale_loss=rationale_loss,
        )
