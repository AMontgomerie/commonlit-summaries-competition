import pandas as pd
import numpy as np


def compute_mcrmse(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    content_rmse = np.sqrt(np.mean((predictions.content - targets.content) ** 2))
    wording_rmse = np.sqrt(np.mean((predictions.wording - targets.wording) ** 2))
    return (content_rmse + wording_rmse) / 2
