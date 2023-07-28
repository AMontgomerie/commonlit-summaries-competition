from __future__ import annotations
import pytest
import torch
from unittest.mock import Mock


@pytest.fixture
def mock_tokenizer() -> Mock:
    class MockTokenizer:
        def __call__(
            self, text1: str, text2: str, truncation: bool, return_tensors: str | None = None
        ) -> dict[str, list[int]]:
            num_tokens = len(text1.split()) + len(text2.split())
            return {
                "input_ids": torch.tensor([i for i in range(num_tokens)]),
                "attention_mask": torch.tensor([1 for _ in range(num_tokens)]),
            }

        def pad(
            self,
            features: list[list[int]],
            padding,
            max_length: int,
            pad_to_multiple_of: bool,
            return_tensors: str | bool,
        ) -> list[int] | torch.Tensor:
            tensor_size = [len(features), max_length]
            return {
                "input_ids": torch.zeros(size=tensor_size),
                "attention_mask": torch.zeros(size=tensor_size),
            }

    return MockTokenizer()


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
