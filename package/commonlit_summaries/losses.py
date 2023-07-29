import torch


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
        loss = 0

        for i in range(self.num_targets):
            loss += self.rmse(predictions[:, i], targets[:, i]) / self.num_targets

        return loss
