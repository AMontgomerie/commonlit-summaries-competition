import numpy as np
import pandas as pd
import typer

from commonlit_summaries.metrics import compute_mcrmse


app = typer.Typer(add_completion=False)


@app.command()
def main(oof_predictions_path: str = typer.Option(..., "--path")):
    predictions = pd.read_csv(oof_predictions_path)
    scores_per_fold = []

    for fold in predictions.prompt_id.unique():
        fold_predictions = predictions.loc[predictions.prompt_id == fold]
        fold_scores = compute_mcrmse(fold_predictions)
        typer.echo(f"Fold {fold} | {fold_scores}")
        scores_per_fold.append(list(fold_scores.values()))

    cv_score = np.mean(scores_per_fold, axis=0)
    typer.echo(f"CV | MCRMSE {cv_score[0]} | Content {cv_score[1]} " f"| Wording {cv_score[2]}")

    oof_scores = compute_mcrmse(predictions)
    typer.echo(f"OOF | {oof_scores}")


if __name__ == "__main__":
    app()
