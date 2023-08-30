import os
import pandas as pd
from pathlib import Path
import tempfile
import torch

from commonlit_summaries.experiment import (
    Experiment,
    RankingExperiment,
    get_lr_scheduler,
    get_optimizer,
)
from commonlit_summaries.losses import get_loss_fn
from commonlit_summaries.models import get_model
from commonlit_summaries.data import (
    PromptType,
    PredictionType,
    SummaryDataset,
    SummaryRankingDataset,
)
from commonlit_summaries.tokenizer import setup_tokenizer
from commonlit_summaries.metrics import get_eval_fn


def test_experiment(mock_data: pd.DataFrame):
    """Creates an `Experiment` and runs a small amount of dummy data through it.

    Models are saved to a temporary directory.
    """
    prediction_type = PredictionType.content
    checkpoint = "distilroberta-base"
    tokenizer = setup_tokenizer(checkpoint)
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
    num_labels = 1
    loss_fn, metrics = get_loss_fn("mse", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler="mean",
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            run_id="test-run",
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
            eval_fn=get_eval_fn(prediction_type),
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
    tokenizer = setup_tokenizer(checkpoint)
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
    num_labels = 2
    loss_fn, metrics = get_loss_fn("mcrmse", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler="max",
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            run_id="test-run",
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
            eval_fn=get_eval_fn(prediction_type),
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs


def test_experiment_no_eval(mock_data: pd.DataFrame):
    """Same as the above test but tests when we use the whole dataset for training, so no eval."""
    prediction_type = PredictionType.both
    checkpoint = "distilroberta-base"
    tokenizer = setup_tokenizer(checkpoint)
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
    num_labels = 2
    loss_fn, metrics = get_loss_fn("mcrmse", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler="mean",
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            run_id="test-run",
            fold="test-fold",
            loss_fn=loss_fn,
            model_name=checkpoint,
            model=model,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            train_dataset=dataset,
            eval_dataset=None,
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
            eval_fn=get_eval_fn(prediction_type),
        )
        model, metrics = experiment.run()

        assert len(os.listdir(tempdir)) == 1

    assert isinstance(model, torch.nn.Module)
    assert not metrics


def test_experiment_both_mcrmse_hfhead(mock_data: pd.DataFrame):
    """Tests predicting both targets with MCRMSE on an AutoModelForSequenceClassification, which
    should be equivalent to adding a linear layer after the main network.
    """
    prediction_type = PredictionType.both
    checkpoint = "distilroberta-base"
    tokenizer = setup_tokenizer(checkpoint)
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
    num_labels = 2
    loss_fn, metrics = get_loss_fn("mcrmse", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        freeze_encoder_layers=3,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler="hf",
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            run_id="test-run",
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
            eval_fn=None,
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs


def test_experiment_both_mcrmse_gemtext(mock_data: pd.DataFrame):
    """Tests predicting both targets with MCRMSE on a model using GemText pooler."""
    prediction_type = PredictionType.both
    checkpoint = "distilroberta-base"
    tokenizer = setup_tokenizer(checkpoint)
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
    num_labels = 2
    loss_fn, metrics = get_loss_fn("mcrmse", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        freeze_embeddings=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pooler="gemtext",
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            run_id="test-run",
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
            eval_fn=get_eval_fn(prediction_type),
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs


def test_experiment_both_mcrmse_attentionhead(mock_data: pd.DataFrame):
    prediction_type = PredictionType.both
    checkpoint = "distilroberta-base"
    tokenizer = setup_tokenizer(checkpoint)
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
    num_labels = 2
    loss_fn, metrics = get_loss_fn("mcrmse", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler="mean",
        use_attention_head=True,
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = Experiment(
            run_id="test-run",
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
            eval_fn=get_eval_fn(prediction_type),
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs


def test_ranking_experiment(mock_data: pd.DataFrame):
    prediction_type = PredictionType.content
    checkpoint = "distilroberta-base"
    tokenizer = setup_tokenizer(checkpoint)
    dataset = SummaryRankingDataset(
        tokenizer,
        mock_data,
        prompt_types=[PromptType.title, PromptType.summary],
        prediction_type=prediction_type,
        fix_length=512,
        seed=666,
    )
    epochs = 2
    batch_size = 4
    accumulation_steps = 2
    num_labels = 1
    loss_fn, metrics = get_loss_fn("ranking", num_labels)
    model = get_model(
        checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pooler="mean",
        use_attention_head=True,
        device="cpu",
    )
    optimizer = get_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    epoch_steps = (len(dataset) // batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(
        "constant", optimizer, warmup_proportion=0, epochs=epochs, steps_per_epoch=epoch_steps
    )

    with tempfile.TemporaryDirectory() as tempdir:
        experiment = RankingExperiment(
            run_id="test-run",
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
            eval_fn=get_eval_fn(prediction_type),
        )
        model, metrics = experiment.run()

        # Check that we have a file output for each epoch
        assert len(os.listdir(tempdir)) == epochs

    assert isinstance(model, torch.nn.Module)
    assert len(metrics) == epochs
