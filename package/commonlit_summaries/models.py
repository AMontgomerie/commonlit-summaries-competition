from dataclasses import dataclass
import torch
from transformers import AutoModel, AutoConfig, PretrainedConfig
from transformers.models.deberta.modeling_deberta import StableDropout, ContextPooler
from typing import Any


class AttentionHead(torch.nn.Module):
    def __init__(self, in_size: int, hidden_size: int = 512):
        super().__init__()
        self.W = torch.nn.Linear(in_size, hidden_size)
        self.V = torch.nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        return context_vector


class MeanPooling(torch.nn.Module):
    """Based on https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3"""

    def __init__(self, config: PretrainedConfig, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
        sum_mask = attention_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=self.epsilon)
        return sum_embeddings / sum_mask


class MaxPooling(torch.nn.Module):
    """Based on https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3"""

    def __init__(self, config: PretrainedConfig, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, last_hidden_state, attention_mask):
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        embeddings = last_hidden_state.clone()
        embeddings[attention_mask_expanded == 0] = -self.epsilon
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class GeMTextPooling(torch.nn.Module):
    """Based on https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3"""

    def __init__(self, config: PretrainedConfig, dim: int = 1, p: int = 3, epsilon: int = 1e-6):
        super().__init__()
        self.dim = dim
        self.p = torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.epsilon = epsilon

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        clamped_hidden_state = last_hidden_state.clamp(min=self.epsilon)
        x = (clamped_hidden_state * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.epsilon)
        return ret.pow(1 / self.p)


class ContextPooling(torch.nn.Module):
    """Wrapper around transformers.models.deberta.modeling_deberta.ContextPooler so it matches our
    other poolers.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.pooler = ContextPooler(config)

    def forward(self, hidden_states: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return self.pooler(hidden_states)


@dataclass
class Output:
    """Wrapper for output logits to match transformers models outputs. Loss and hidden states not
    implemented as unused.
    """

    logits: torch.Tensor


class CommonlitRegressorModel(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str,
        num_labels: int,
        pooler_cls: type[torch.nn.Module],
        use_attention_head: bool = False,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(checkpoint, **kwargs)
        self.transformer = AutoModel.from_pretrained(checkpoint, config=config)
        self.pooler = pooler_cls(config)
        self.regressor = torch.nn.Linear(config.hidden_size, num_labels)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.use_attention_head = use_attention_head

        if self.use_attention_head:
            self.attention_head = AttentionHead(
                in_size=config.hidden_size, hidden_size=config.hidden_size
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Output:
        transformer_output = self.transformer(input_ids, attention_mask, output_hidden_states=False)
        sequence_output = transformer_output.last_hidden_state

        if self.use_attention_head:
            sequence_output = self.attention_head(sequence_output)

        pooled_output = self.pooler(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        regressor_output = self.regressor(pooled_output)
        return Output(regressor_output)

    def resize_token_embeddings(self, size: int) -> None:
        self.transformer.resize_token_embeddings(size)
