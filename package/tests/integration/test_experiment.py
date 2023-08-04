import os
import pandas as pd
from pathlib import Path
import tempfile
import torch
from transformers import AutoTokenizer

from commonlit_summaries.train import get_loss_fn, get_lr_scheduler, get_model, get_optimizer
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
        tokenizer,
        mock_data,
        prompt_types=[PromptType.summary, PromptType.question, PromptType.text],
        prediction_type=prediction_type,
        fix_length=512,
    )
    epochs = 2
    batch_size = 4
    accumulation_steps = 1
    loss_fn, metrics = get_loss_fn("mse")
    model = get_model(checkpoint, prediction_type, device="cpu")
    optimizer = get_optimizer(model, learning_rate=1e-5)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            fold="test-fold",
            metrics=metrics,
            loss_fn=loss_fn,
            model_name=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            train_dataset=dataset,
            eval_dataset=dataset,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            device="cpu",
            epochs=epochs,
            save_dir=Path(tempdir),
            save_strategy="last",
            accumulation_steps=2,
            dataloader_workers=0,
            use_wandb=False,
            log_interval=100,
        )
        model, metrics = experiment.run()

        assert len(os.listdir(tempdir)) == 1

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
        tokenizer,
        mock_data,
        prompt_types=[PromptType.title, PromptType.summary],
        prediction_type=prediction_type,
        fix_length=512,
    )
    epochs = 2
    batch_size = 4
    accumulation_steps = 2
    loss_fn, metrics = get_loss_fn("mcrmse")
    model = get_model(checkpoint, prediction_type, device="cpu")
    optimizer = get_optimizer(model, learning_rate=1e-5)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            fold="test-fold",
            loss_fn=loss_fn,
            model_name=checkpoint,
            model=model,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            train_dataset=dataset,
            eval_dataset=dataset,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            device="cpu",
            epochs=epochs,
            save_dir=Path(tempdir),
            save_strategy="all",
            accumulation_steps=2,
            dataloader_workers=0,
            use_wandb=False,
            log_interval=100,
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs
