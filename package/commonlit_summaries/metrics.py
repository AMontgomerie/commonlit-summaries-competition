import pandas as pd
import numpy as np


def compute_metrics(data: pd.DataFrame) -> tuple[float, float, float]:
    content_rmse = np.sqrt(np.mean((data.content - data.predicted_content) ** 2))
    wording_rmse = np.sqrt(np.mean((data.wording - data.predicted_wording) ** 2))
    mcrmse = np.mean([content_rmse, wording_rmse])
    return {"MCRMSE": mcrmse, "ContentRMSE": content_rmse, "WordingRMSE": wording_rmse}


def compute_metrics_np(targets: np.array, predictions: np.array) -> tuple[float, float, float]:
    content_rmse = np.sqrt(np.mean((targets[:, 0] - predictions[:, 0]) ** 2))
    wording_rmse = np.sqrt(np.mean((targets[:, 1] - predictions[:, 1]) ** 2))
    mcrmse = np.mean([content_rmse, wording_rmse])
    return {"MCRMSE": mcrmse, "ContentRMSE": content_rmse, "WordingRMSE": wording_rmse}
