import pandas as pd
from pathlib import Path
import pytest
from pytest_mock import MockerFixture

from commonlit_summaries.inference import Model
from commonlit_summaries.data import load_data, PromptType

DATA_DIR = Path(__file__).parents[3] / "data"


@pytest.fixture
def train_data() -> pd.DataFrame:
    return load_data(DATA_DIR, train=True)


def test_predict(
    mocker: MockerFixture, mock_model_for_sequence_classification, train_data: pd.DataFrame
):
    mocker.patch(
        "commonlit_summaries.inference.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model_for_sequence_classification,
    )
    model = Model(checkpoint="distilroberta-base", max_length=512, num_labels=2, device="cpu")
    predictions = model.predict(
        train_data,
        batch_size=128,
        dataloader_num_workers=0,
        prompt_types=[PromptType.summary, PromptType.question],
    )
    assert len(predictions) == len(train_data)
