from dataclasses import dataclass
import torch
from transformers import AutoModel, AutoConfig


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


class CommonlitRegressorModel(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str,
        num_labels: int,
        use_pooler: bool = False,
        use_attention_head: bool = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(checkpoint)
        self.transformer = AutoModel.from_pretrained(checkpoint, config=config)
        self.regressor = torch.nn.Linear(config.hidden_size, num_labels)
        self.use_pooler = use_pooler
        self.use_attention_head = use_attention_head

        if self.use_pooler:
            self.pooler = GeMText()

        if self.use_attention_head:
            self.attention_head = AttentionHead(
                in_size=config.hidden_size, hidden_size=config.hidden_size
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Output:
        transformer_output = self.transformer(input_ids, attention_mask, output_hidden_states=False)
        output = transformer_output.last_hidden_state

        if self.use_pooler:
            output = self.pooler(output, attention_mask)

        if self.use_attention_head:
            output = self.attention_head(output)

        regressor_output = self.regressor(output)
        return Output(regressor_output)

    def resize_token_embeddings(self, size: int) -> None:
        self.transformer.resize_token_embeddings(size)
