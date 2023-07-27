import numpy as np
import os
import pandas as pd
from pathlib import Path
import typer

from data import load_data, PredictionType
from inference import Model

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    weights_dir: Path = typer.Option(..., "--weights"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    output_dir: Path = typer.Option("./", "--output-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    batch_size: int = typer.Option(32, "--api-key"),
):
    data = load_data(data_dir, train=False)
    model = Model(model_checkpoint, max_length)
    all_predictions = {"content": [], "wording": []}

    for filename in os.listdir(weights_dir):
        path = weights_dir / filename
        model.load_weights(path)
        prediction_type = PredictionType.content if "content" in filename else "wording"
        predictions = model.predict(data, prediction_type, batch_size)
        all_predictions[prediction_type].append(predictions)

    mean_content_predictions = np.mean(all_predictions["content"], axis=0)
    mean_wording_predictions = np.mean(all_predictions["wording"], axis=0)
    predictions_df = pd.DataFrame(
        {
            "student_id": data.student_id,
            "content": mean_content_predictions,
            "wording": mean_wording_predictions,
        }
    )
    output_filename = model_checkpoint.replace("/", "-")
    predictions_df.to_csv(output_dir / output_filename, index=False)


if __name__ == "__main__":
    app()
