import pandas as pd
import numpy as np
from typing import Callable

from commonlit_summaries.data import PredictionType


def compute_mcrmse(data: pd.DataFrame) -> dict[str, float]:
    content_rmse = np.sqrt(np.mean((data.content - data.predicted_content) ** 2))
    wording_rmse = np.sqrt(np.mean((data.wording - data.predicted_wording) ** 2))
    mcrmse = np.mean([content_rmse, wording_rmse])
    return {"MCRMSE": mcrmse, "ContentRMSE": content_rmse, "WordingRMSE": wording_rmse}


def compute_mcrmse_np(targets: np.array, predictions: np.array) -> dict[str, float]:
    content_rmse = np.sqrt(np.mean((targets[:, 0] - predictions[:, 0]) ** 2))
    wording_rmse = np.sqrt(np.mean((targets[:, 1] - predictions[:, 1]) ** 2))
    mcrmse = np.mean([content_rmse, wording_rmse])
    return {"MCRMSE": mcrmse, "ContentRMSE": content_rmse, "WordingRMSE": wording_rmse}


def compute_rmse_np(targets: np.array, predictions: np.array) -> dict[str, float]:
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    return {"RMSE": rmse}


def get_eval_fn(prediction_type: PredictionType) -> Callable:
    """Returns a function which computes RMSE for single target predictions, and MCRMSE for double
    target predictions.
    """
    if prediction_type == PredictionType.both:
        return compute_mcrmse_np
    else:
        return compute_rmse_np
