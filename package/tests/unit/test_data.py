import pandas as pd
from pathlib import Path
import pytest
from unittest.mock import Mock

from commonlit_summaries.data import SummaryDataset

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def mock_tokenizer() -> Mock:
    def mock_tokenize(text1: str, text2: str, truncation: bool) -> dict[str, list[int]]:
        num_tokens = len(text1.split()) + len(text2.split())
        return {
            "input_ids": [i for i in range(num_tokens)],
            "attention_mask": [1 for _ in range(num_tokens)],
        }

    tokenizer = Mock
    tokenizer.__call__ = mock_tokenize


@pytest.fixture
def mock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": [1, 2, 3],
            "prompt_id": [1, 2, 3],
            "prompt_question": ["a", "b", "c"],
            "text": ["a", "b", "c"],
            "content": [0, 0.5, 1],
            "wording": [0, 0.5, 1],
        }
    )


def test_create_inference_dataset(mock_tokenizer: Mock, mock_data: pd.DataFrame):
    dataset = SummaryDataset(mock_tokenizer, mock_data, train=False)
    inputs = dataset[0]
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "labels" not in inputs
    assert isinstance(inputs["input_ids"], list)
    assert isinstance(inputs["attention_mask"], list)
    assert len(inputs["input_ids"]) == len(inputs["attention_mask"])


def test_create_train_dataset(mock_tokenizer: Mock, mock_data: pd.DataFrame):
    dataset = SummaryDataset(mock_tokenizer, mock_data)
    inputs = dataset[0]
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "labels" in inputs
    assert isinstance(inputs["input_ids"], list)
    assert isinstance(inputs["attention_mask"], list)
    assert len(inputs["input_ids"]) == len(inputs["attention_mask"])
