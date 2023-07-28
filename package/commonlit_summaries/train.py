from pathlib import Path
from transformers import AutoTokenizer
import typer

from commonlit_summaries.data import PredictionType, SummaryDataset, load_data
from commonlit_summaries.trainer import Trainer
from commonlit_summaries.utils import set_seed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prediction_type: PredictionType = typer.Option(..., "--prediction-type"),
    fold: str = typer.Option(..., "--fold"),
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    train_batch_size: int = typer.Option(32, "--train-batch-size"),
    valid_batch_size: int = typer.Option(128, "--valid-batch-size"),
    epochs: int = typer.Option(1, "--epochs"),
    seed: int = typer.Option(666, "--seed"),
    scheduler: str = typer.Option("constant", "--scheduler"),
    warmup: float = typer.Option(0.0, "--warmup"),
):
    set_seed(seed)
    data = load_data(data_dir)
    train_data = data.loc[data.prompt_id != fold].reset_index(drop=True)
    valid_data = data.loc[data.prompt_id == fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = SummaryDataset(tokenizer, train_data, prediction_type, fix_length=max_length)
    valid_dataset = SummaryDataset(tokenizer, valid_data, prediction_type, fix_length=max_length)
    trainer = Trainer(
        prediction_type=prediction_type,
        fold=fold,
        model_checkpoint=model_checkpoint,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        max_length=max_length,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=valid_batch_size,
        scheduler=scheduler,
        warmup=warmup,
    )
    trainer.train(epochs)


if __name__ == "__main__":
    app()
