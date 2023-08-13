from dataclasses import dataclass
import torch
from transformers import AutoModel, AutoConfig


class GeMText(torch.nn.Module):
    """Based on https://www.kaggle.com/code/asteyagaur/commonlit-deberta-v3"""

    def __init__(self, dim: int = 1, p: int = 3, epsilon: int = 1e-6):
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


@dataclass
class Output:
    """Wrapper for output logits to match transformers models outputs. Loss and hidden states not
    implemented as unused.
    """

    logits: torch.Tensor


class GemTextPoolerModel(torch.nn.Module):
    """A transformer model with a GemText pooling layer before the regression head."""

    def __init__(self, checkpoint: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(checkpoint)
        self.transformer = AutoModel.from_pretrained(checkpoint, config=config)
        self.pooler = GeMText()
        self.regressor = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Output:
        transformer_output = self.transformer(input_ids, attention_mask, output_hidden_states=False)
        pooled_output = self.pooler(transformer_output.last_hidden_state, attention_mask)
        regressor_output = self.regressor(pooled_output)
        return Output(regressor_output)
