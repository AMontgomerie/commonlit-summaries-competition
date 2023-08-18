import numpy as np
import pandas as pd
import typer

from commonlit_summaries.metrics import compute_metrics


app = typer.Typer(add_completion=False)


@app.command()
def main(oof_predictions_path: str = typer.Option(..., "--path")):
    predictions = pd.read_csv(oof_predictions_path)
    fold_scores = []

    for fold in predictions.prompt_id.unique():
        fold_predictions = predictions.loc[predictions.prompt_id == fold]
        fold_mcrmse, fold_content_rmse, fold_wording_rmse = compute_metrics(fold_predictions)
        typer.echo(
            f"Fold {fold} | MCRMSE {fold_mcrmse} | Content {fold_content_rmse} "
            f"| Wording {fold_wording_rmse}"
        )
        fold_scores.append([fold_mcrmse, fold_content_rmse, fold_wording_rmse])

    cv_score = np.mean(fold_scores, axis=0)
    typer.echo(f"CV | MCRMSE {cv_score[0]} | Content {cv_score[1]} " f"| Wording {cv_score[2]}")

    oof_mcrmse, oof_content_rmse, oof_wording_rmse = compute_metrics(predictions)
    typer.echo(
        f"CV | MCRMSE {oof_mcrmse} | Content {oof_content_rmse} " f"| Wording {oof_wording_rmse}"
    )


if __name__ == "__main__":
    app()
