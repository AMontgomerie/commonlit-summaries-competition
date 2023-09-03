import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer

from commonlit_summaries.data import (
    SummaryDataset,
    SummaryRankingDataset,
    PromptType,
    PredictionType,
    load_data,
)

DATA_DIR = Path(__file__).parents[3] / "data"


def test_create_inference_dataset(tokenizer: AutoTokenizer, mock_data: pd.DataFrame):
    length = 512
    dataset = SummaryDataset(
        tokenizer,
        mock_data,
        prompt_types=[PromptType.summary, PromptType.question],
        max_length=length,
        pad=True,
    )
    inputs = dataset[0]
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "labels" not in inputs
    assert isinstance(inputs["input_ids"], torch.Tensor)
    assert isinstance(inputs["attention_mask"], torch.Tensor)
    assert len(inputs["input_ids"]) == len(inputs["attention_mask"]) == length


def test_create_train_dataset(tokenizer: AutoTokenizer, mock_data: pd.DataFrame):
    dataset = SummaryDataset(
        tokenizer,
        mock_data,
        prompt_types=[PromptType.summary],
        prediction_type=PredictionType.content,
        pad=True,
    )
    inputs = dataset[0]
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "labels" in inputs
    assert isinstance(inputs["input_ids"], torch.Tensor)
    assert isinstance(inputs["attention_mask"], torch.Tensor)
    assert len(inputs["input_ids"]) == len(inputs["attention_mask"])


def test_create_ranking_train_dataset(tokenizer: AutoTokenizer, mock_data: pd.DataFrame):
    dataset = SummaryRankingDataset(
        tokenizer,
        mock_data,
        prompt_types=[PromptType.summary],
        prediction_type=PredictionType.content,
        seed=666,
        pad=True,
    )
    input1, input2, labels = dataset[0]
    assert "input_ids" in input1
    assert "attention_mask" in input1
    assert "labels" in input1
    assert "input_ids" in input2
    assert "attention_mask" in input2
    assert "labels" in input2
    assert all(labels <= 1) and all(labels >= -1)
    assert not input1["input_ids"].equal(input2["input_ids"])


def test_load_test_data():
    data = load_data(DATA_DIR, train=False)

    for column in ["student_id", "prompt_id", "prompt_question", "prompt_text", "text"]:
        assert column in data.columns

    for column in ["content", "wording"]:
        assert column not in data.columns

    assert len(data) > 0


def test_load_train_data():
    data = load_data(DATA_DIR, train=True)
    for column in [
        "student_id",
        "prompt_id",
        "prompt_question",
        "prompt_text",
        "text",
        "content",
        "wording",
    ]:
        assert column in data.columns

    assert len(data) > 0
