import os
import pandas as pd
from pathlib import Path
import tempfile
import torch
from transformers import AutoTokenizer

from commonlit_summaries.experiment import Experiment
from commonlit_summaries.data import PromptType, PredictionType, SummaryDataset


def test_experiment(mock_data: pd.DataFrame):
    """Creates an `Experiment` and runs a small amount of dummy data through it.

    Models are saved to a temporary directory.
    """
    prediction_type = PredictionType.content
    checkpoint = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = SummaryDataset(
        tokenizer, mock_data, PromptType.both, prediction_type=prediction_type, fix_length=512
    )
    epochs = 2

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            prediction_type=prediction_type,
            fold="test-fold",
            model_name=checkpoint,
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
            loss="mse",
            accumulation_steps=2,
            dataloader_workers=0,
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs


def test_experiment_both_mcrmse(mock_data: pd.DataFrame):
    """Same as the above test but tests predicting on both content and wording at the same time
    with MCRMSE loss.
    """
    prediction_type = PredictionType.both
    checkpoint = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = SummaryDataset(
        tokenizer, mock_data, PromptType.question, prediction_type=prediction_type, fix_length=512
    )
    epochs = 2

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            prediction_type=prediction_type,
            fold="test-fold",
            model_name=checkpoint,
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
            loss="mcrmse",
            accumulation_steps=1,
            dataloader_workers=0,
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs
