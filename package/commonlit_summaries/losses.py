import torch
from torch.nn import MSELoss, SmoothL1Loss, HuberLoss


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


class TiedMarginRankingLoss(torch.nn.Module):
    """Modified `torch.nn.MarginRankingLoss` which allows for ties.

    Targets should be 0 when input1 == input2, 1 when input1 > input2, and -1 otherwise. The margin
    is only applied to non-tied examples.

    Only mean reduction is currently supported.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        elementwise_loss = torch.where(
            target == 0,
            torch.abs(input1 - input2),
            torch.clamp_min(-target * (input1 - input2) + self.margin, 0),
        )
        return torch.mean(elementwise_loss)


def get_loss_fn(
    name: str, num_labels: int, loss_threshold: float
) -> tuple[torch.nn.Module, list[str]]:
    losses = {
        "mse": (MSELoss, ["MSE"]),
        "rmse": (RMSELoss, ["RMSE"]),
        "mcrmse": (MCRMSELoss, ["MCRMSE", "C", "W"]),
        "ranking": (TiedMarginRankingLoss, ["Ranking"]),
        "smoothl1": (SmoothL1Loss, ["SmoothL1"]),
        "huber": (HuberLoss, ["Huber"]),
    }

    if name not in losses:
        raise ValueError(f"{name} is not a valid loss function.")

    loss_fn, metrics = losses[name]

    if num_labels > 1 and name in ["mse", "rmse", "smoothl1", "huber"]:
        if name == "huber":
            criterion = loss_fn(reduction="mean", delta=loss_threshold)
        elif name == "smoothl1":
            criterion = loss_fn(reduction="mean", beta=loss_threshold)
        else:
            criterion = loss_fn(reduction="mean")

    elif name == "ranking":
        criterion = loss_fn(margin=loss_threshold)
    else:
        criterion = loss_fn()

    return criterion, metrics
