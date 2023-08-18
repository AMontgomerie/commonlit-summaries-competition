import torch
from torch.nn import MSELoss, MarginRankingLoss, SmoothL1Loss


class RMSELoss(torch.nn.Module):
    """Based on https://www.kaggle.com/code/masashisode/pytorch-implementation-of-mcrmseloss"""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.mse(predictions, targets) + self.epsilon)
        return loss


class MCRMSELoss(torch.nn.Module):
    """Based on https://www.kaggle.com/code/masashisode/pytorch-implementation-of-mcrmseloss"""

    def __init__(self, num_targets: int = 2):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_targets = num_targets

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        rmse_per_column = []

        for column in range(self.num_targets):
            rmse = self.rmse(predictions[:, column], targets[:, column])
            rmse_per_column.append(rmse)

        mcrmse = sum(rmse_per_column) / self.num_targets

        return mcrmse, *rmse_per_column


def get_loss_fn(name: str, num_labels: int) -> tuple[torch.nn.Module, list[str]]:
    losses = {
        "mse": (MSELoss, ["MSE"]),
        "rmse": (RMSELoss, ["RMSE"]),
        "mcrmse": (MCRMSELoss, ["MCRMSE", "C", "W"]),
        "ranking": (MarginRankingLoss),
        "smoothl1": (SmoothL1Loss, ["SmoothL1"]),
    }

    if name not in losses:
        raise ValueError(f"{name} is not a valid loss function.")

    loss_fn, metrics = losses[name]

    if num_labels > 1 and name in ["mse", "rmse", "smoothl1"]:
        criterion = loss_fn(reduction="mean")
    else:
        criterion = loss_fn()

    return criterion, metrics
