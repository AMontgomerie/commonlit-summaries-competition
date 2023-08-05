from pathlib import Path
import typer

from commonlit_summaries.data import load_data, PromptType
from commonlit_summaries.inference import Model

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prompt_types: list[PromptType] = typer.Option(..., "--prompt-type"),
    model_name: str = typer.Option(..., "--name"),
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    weights_dir: Path = typer.Option(..., "--weights-dir"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    output_dir: Path = typer.Option("./", "--output-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    batch_size: int = typer.Option(32, "--batch-size"),
):
    data = load_data(data_dir, train=True)
    model = Model(model_checkpoint, max_length, num_labels=2)

    for fold in data.prompt_id.unique():
        path = get_weights_file_path(fold, weights_dir)
        model.load_weights(path)
        fold_data = data.loc[data.prompt_id == fold]
        predictions = model.predict(fold_data, batch_size, prompt_types)
        fold_data["predicted_content"] = predictions[:, 0]
        fold_data["predicted_wording"] = predictions[:, 1]

    data.to_csv(output_dir / f"{model_name}-oof.csv", index=False)


def get_weights_file_path(fold: str, weights_dir: Path) -> Path:
    for filename in weights_dir.iterdir():
        if fold in str(filename) and filename.suffix == ".bin":
            return filename.absolute()

    raise FileNotFoundError(f"Couldn't find a weights file for fold {fold}.")


if __name__ == "__main__":
    app()
