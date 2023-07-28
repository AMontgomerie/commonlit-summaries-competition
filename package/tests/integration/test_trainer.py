import os
import pandas as pd
from pathlib import Path
import tempfile
import torch
from transformers import AutoTokenizer

from commonlit_summaries.trainer import Trainer
from commonlit_summaries.data import PredictionType, SummaryDataset


def test_trainer(mock_data: pd.DataFrame):
    """Creates a `Trainer` and runs a small amount of dummy data through it.

    Models are saved to a temporary directory.
    """
    prediction_type = PredictionType.content
    checkpoint = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = SummaryDataset(tokenizer, mock_data, prediction_type=prediction_type, fix_length=512)
    epochs = 2

    with tempfile.TemporaryDirectory() as tempdir:
        trainer = Trainer(
            prediction_type=prediction_type,
            fold="test-fold",
            model_checkpoint=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
            learning_rate=1e-5,
            train_batch_size=4,
            eval_batch_size=4,
            scheduler="constant",
            warmup=0.0,
            device="cpu",
            epochs=epochs,
            save_dir=Path(tempdir),
        )
        model, metrics = trainer.train()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs
