from pathlib import Path
import torch
from torch.nn import MSELoss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import typer

from commonlit_summaries.data import PromptType, PredictionType, SummaryDataset, load_data
from commonlit_summaries.experiment import Experiment
from commonlit_summaries.utils import set_seed
from commonlit_summaries.losses import RMSELoss, MCRMSELoss

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prompt_type: PromptType = typer.Option(..., "--prompt-type"),
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
    accumulation_steps: int = typer.Option(1, "--accumulation-steps"),
    loss: str = typer.Option("mse", "--loss"),
):
    device = "cuda"
    set_seed(seed)
    data = load_data(data_dir)
    train_data = data.loc[data.prompt_id != fold].reset_index(drop=True)
    valid_data = data.loc[data.prompt_id == fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = SummaryDataset(
        tokenizer, train_data, prompt_type, prediction_type, fix_length=max_length
    )
    valid_dataset = SummaryDataset(
        tokenizer, valid_data, prompt_type, prediction_type, fix_length=max_length
    )
    loss_fn = get_loss_fn(loss)
    model = get_model(model_checkpoint, prediction_type, device)
    optimizer = get_optimizer(model, learning_rate)
    epoch_steps = (len(train_dataset) // train_batch_size) // accumulation_steps
    lr_scheduler = get_lr_scheduler(scheduler_name, optimizer, warmup, epochs, epoch_steps)
    experiment = Experiment(
        fold=fold,
        loss_fn=loss_fn,
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=valid_batch_size,
        epochs=epochs,
        save_dir=save_dir,
        accumulation_steps=accumulation_steps,
    )
    experiment.run()


def get_model(
    model_checkpoint: str, prediction_type: PredictionType, device: str
) -> AutoModelForSequenceClassification:
    num_labels = 2 if prediction_type == PredictionType.both else 1
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )
    return model.to(device)


def get_loss_fn(name: str) -> torch.nn.Module:
    losses = {"mse": MSELoss, "rmse": RMSELoss, "mcrmse": MCRMSELoss}

    if name not in losses:
        raise ValueError(f"{name} is not a valid loss function.")

    return losses[name]()


def get_optimizer(model: AutoModelForSequenceClassification, learning_rate: float) -> Optimizer:
    return AdamW(model.parameters(), lr=learning_rate)


def get_lr_scheduler(
    scheduler_name: str, optimizer, warmup_proportion: float, epochs: int, steps_per_epoch: int
) -> LRScheduler:
    total_steps = steps_per_epoch * epochs
    num_warmup_steps = round(total_steps * warmup_proportion)
    return get_scheduler(scheduler_name, optimizer, num_warmup_steps, total_steps)


if __name__ == "__main__":
    app()
