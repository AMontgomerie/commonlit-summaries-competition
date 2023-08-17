from pathlib import Path
import torch
from torch.nn import MSELoss, MarginRankingLoss, SmoothL1Loss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForSequenceClassification, get_scheduler
import typer
import wandb

from commonlit_summaries.data import PromptType, PredictionType, SummaryDataset, load_data
from commonlit_summaries.experiment import Experiment
from commonlit_summaries.utils import set_seed
from commonlit_summaries.losses import RMSELoss, MCRMSELoss
from commonlit_summaries.tokenizer import setup_tokenizer
from commonlit_summaries.models import (
    CommonlitRegressorModel,
    MeanPooling,
    MaxPooling,
    GeMTextPooling,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    group_id: int = typer.Option(..., "--group-id"),
    prompt_types: list[PromptType] = typer.Option(..., "--prompt-type"),
    prediction_type: PredictionType = typer.Option(..., "--prediction-type"),
    fold: str = typer.Option(..., "--fold"),
    model_name: str = typer.Option(..., "--name"),
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    train_batch_size: int = typer.Option(32, "--train-batch-size"),
    valid_batch_size: int = typer.Option(128, "--valid-batch-size"),
    epochs: int = typer.Option(1, "--epochs"),
    seed: int = typer.Option(666, "--seed"),
    scheduler_name: str = typer.Option("constant", "--scheduler"),
    warmup: float = typer.Option(0.0, "--warmup"),
    save_dir: Path = typer.Option("./", "--save-dir"),
    save_strategy: str = typer.Option("last", "--save-strategy"),
    accumulation_steps: int = typer.Option(1, "--accumulation-steps"),
    loss: str = typer.Option("mse", "--loss"),
    log_interval: int = typer.Option(100, "--log-interval"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    summariser_checkpoint: str = typer.Option("facebook/bart-large-cnn", "--summariser-checkpoint"),
    summariser_max_length: int = typer.Option(1024, "--summariser-max-length"),
    summariser_min_length: int = typer.Option(1024, "--summariser-min-length"),
    pooler: str = typer.Option("mean", "--pooler"),
    use_attention_head: bool = typer.Option(False, "--use-attention-head"),
    dropout: bool = typer.Option(0.1, "--dropout"),
):
    wandb.login()
    wandb.init(
        project="commonlit-summaries",
        group=str(group_id),
        name=f"{group_id}-{model_name}-{fold}",
        config=locals(),
    )
    set_seed(seed)

    data = load_data(
        data_dir,
        train=True,
        summarise=PromptType.reference_summary in prompt_types,
        checkpoint=summariser_checkpoint,
        max_length=summariser_max_length,
        min_length=summariser_min_length,
    )

    print(f"Configuring inputs as: {[p.value for p in prompt_types]}")
    tokenizer = setup_tokenizer(model_checkpoint)

    if fold == "all":
        train_dataset = SummaryDataset(
            tokenizer, data, prompt_types, prediction_type, fix_length=max_length
        )
        valid_dataset = None
    else:
        train_data = data.loc[data.prompt_id != fold].reset_index(drop=True)
        valid_data = data.loc[data.prompt_id == fold].reset_index(drop=True)
        train_dataset = SummaryDataset(
            tokenizer, train_data, prompt_types, prediction_type, fix_length=max_length
        )
        valid_dataset = SummaryDataset(
            tokenizer, valid_data, prompt_types, prediction_type, fix_length=max_length
        )

    num_labels = 2 if prediction_type == PredictionType.both else 1
    loss_fn, metrics = get_loss_fn(loss, num_labels)
    model = get_model(
        model_checkpoint,
        num_labels,
        tokenizer_embedding_size=len(tokenizer),
        drpoout=dropout,
        pooler=pooler,
        use_attention_head=use_attention_head,
        device="cuda",
    )
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    epoch_steps = (len(train_dataset) // train_batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(scheduler_name, optimizer, warmup, epochs, epoch_steps)
    experiment = Experiment(
        run_id=group_id,
        fold=fold,
        loss_fn=loss_fn,
        model_name=model_name,
        model=model,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        train_batch_size=train_batch_size,
        eval_batch_size=valid_batch_size,
        epochs=epochs,
        save_dir=save_dir,
        accumulation_steps=accumulation_steps,
        save_strategy=save_strategy,
        log_interval=log_interval,
        use_wandb=True,
    )
    experiment.run()


def get_model(
    model_checkpoint: str,
    num_labels: int,
    tokenizer_embedding_size: int,
    dropout: float,
    pooler: str,
    use_attention_head: bool = False,
    device: str = "cuda",
) -> AutoModelForSequenceClassification:
    if pooler == "hf":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels, hidden_dropout_prob=dropout
        )
    else:
        pooler_layer = _get_pooling_layer(pooler)
        model = CommonlitRegressorModel(
            model_checkpoint, num_labels, pooler_layer, use_attention_head
        )

    model.resize_token_embeddings(tokenizer_embedding_size)
    return model.to(device)


def _get_pooling_layer(pooler_name: str) -> torch.nn.Module:
    pooling_layers = {"mean": MeanPooling, "max": MaxPooling, "gemtext": GeMTextPooling}
    if pooler_name not in pooling_layers:
        raise ValueError(f"Unknown pooling layer {pooler_name}.")

    return pooling_layers[pooler_name]()


def get_loss_fn(name: str, num_labels: int) -> tuple[torch.nn.Module, list[str]]:
    losses = {
        "mse": (MSELoss, ["MSE"]),
        "rmse": (RMSELoss, ["RMSE"]),
        "mcrmse": (MCRMSELoss, ["MCRMSE", "C", "W"]),
        "ranking": (MarginRankingLoss),
        "smoothl1": (SmoothL1Loss, ["SmoothL1"]),
    }

    if name not in losses:
        raise ValueError(f"{name} is not a valid loss function.")

    loss_fn, metrics = losses[name]

    if num_labels > 1 and name in ["mse", "rmse", "smoothl1"]:
        criterion = loss_fn(reduction="mean")
    else:
        criterion = loss_fn()

    return criterion, metrics


def get_optimizer(
    model: AutoModelForSequenceClassification, learning_rate: float, weight_decay: float
) -> Optimizer:
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_lr_scheduler(
    scheduler_name: str, optimizer, warmup_proportion: float, epochs: int, steps_per_epoch: int
) -> LRScheduler:
    total_steps = steps_per_epoch * epochs
    num_warmup_steps = round(total_steps * warmup_proportion)
    return get_scheduler(scheduler_name, optimizer, num_warmup_steps, total_steps)


if __name__ == "__main__":
    app()
