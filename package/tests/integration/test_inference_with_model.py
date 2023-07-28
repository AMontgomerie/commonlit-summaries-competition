import pandas as pd
from pathlib import Path
import pytest
import torch

from commonlit_summaries.inference import Model
from commonlit_summaries.data import load_data

DATA_DIR = Path(__file__).parents[3] / "data"


@pytest.fixture
def model():
    return Model(checkpoint="distilroberta-base", max_length=512, device="cpu")


@pytest.fixture
def data() -> pd.DataFrame:
    # Sample some rows from the training set since they contain real data
    data = load_data(DATA_DIR, train=True)
    return data.sample(20).reset_index(drop=True)


def test_predict(model: Model, data: pd.DataFrame):
    predictions = model.predict(data, batch_size=4, dataloader_num_workers=0)
    assert len(predictions) == len(data)


def test_load_weights(model: Model):
    weights_path = DATA_DIR / "distilroberta-base-weights.bin"
    old_state_dict = model.model.state_dict()
    model.load_weights(weights_path)

    # Check that at least some of the model parameters are different now.
    new_state_dict = model.model.state_dict()
    assert any(torch.equal(n, o) for n, o in zip(new_state_dict.values(), old_state_dict.values()))
