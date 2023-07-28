import pandas as pd
from pathlib import Path
import pytest
from pytest_mock import MockerFixture

from commonlit_summaries.inference import Model
from commonlit_summaries.data import load_data

DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.fixture
def model(mocker: MockerFixture, mock_tokenizer, mock_model_for_sequence_classification) -> Model:
    mocker.patch(
        "commonlit_summaries.inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    )
    mocker.patch(
        "commonlit_summaries.inference.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model_for_sequence_classification,
    )
    return Model(checkpoint="distilroberta-base", max_length=512, device="cpu")


@pytest.fixture
def train_data() -> pd.DataFrame:
    return load_data(DATA_DIR, train=True)


def test_predict(model: Model, train_data: pd.DataFrame):
    predictions = model.predict(train_data, batch_size=128, dataloader_num_workers=0)
    assert len(predictions) == len(train_data)
