from pathlib import Path
from transformers import AutoTokenizer
import typer

from data import PredictionType, SummaryDataset, load_data
from trainer import Trainer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prediction_type: PredictionType = typer.Option(..., "--prediction-type"),
    fold: str = typer.Option(..., "--fold"),
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    learning_rate: float = typer.Option(1e-5, "--api-key"),
    train_batch_size: int = typer.Option(32, "--api-key"),
    valid_batch_size: int = typer.Option(128, "--api-key"),
    epochs: int = typer.Option(1, "--epochs"),
):
    data = load_data(data_dir)
    train_data = data.loc[data.prompt_id != fold]
    valid_data = data.loc[data.prompt_id == fold]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = SummaryDataset(tokenizer, train_data, prediction_type)
    valid_dataset = SummaryDataset(tokenizer, valid_data, prediction_type)
    trainer = Trainer(
        fold=fold,
        model_checkpoint=model_checkpoint,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        max_length=max_length,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
    )
    trainer.train(epochs)


if __name__ == "__main__":
    app()
