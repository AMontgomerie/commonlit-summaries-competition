from __future__ import annotations
import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import Mock


@pytest.fixture
def tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("distilroberta-base")


@pytest.fixture
def mock_model_for_sequence_classification():
    class MockModel:
        def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            output = Mock()
            output.logits = torch.tensor([0.5] * input_ids.shape[0]).unsqueeze(dim=0)
            output.loss = torch.tensor([0.5]).unsqueeze(dim=0)
            return output

        def to(self, device: str) -> MockModel:
            return self

        def eval(self):
            pass

        def train(self):
            pass

    return MockModel()
