import numpy as np
import os
import pandas as pd
from pathlib import Path
import typer

from commonlit_summaries.data import load_data
from commonlit_summaries.inference import Model

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_checkpoint: str = typer.Option(..., "--checkpoint"),
    weights_dir: Path = typer.Option(..., "--weights-dir"),
    data_dir: Path = typer.Option("./", "--data-dir"),
    output_dir: Path = typer.Option("./", "--output-dir"),
    max_length: int = typer.Option(512, "--max-length"),
    batch_size: int = typer.Option(32, "--batch-size"),
):
    data = load_data(data_dir, train=False)
    model = Model(model_checkpoint, max_length)
    all_predictions = {"content": [], "wording": []}
    model_weights_files = [f for f in os.listdir(weights_dir) if f.endswith(".bin")]

    for filename in model_weights_files:
        path = weights_dir / filename
        model.load_weights(path)
        predictions = model.predict(data, batch_size)
        prediction_type = "content" if "content" in filename else "wording"
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
    predictions_df.to_csv(output_dir / f"{output_filename}-submission.csv", index=False)


if __name__ == "__main__":
    app()