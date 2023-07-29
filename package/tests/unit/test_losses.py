import torch

from commonlit_summaries.losses import RMSELoss, MCRMSELoss


def test_rmse_loss():
    loss_fn = RMSELoss()
    predictions = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float32)
    loss = loss_fn(predictions, targets)
    assert loss < 0.1

    predictions = torch.tensor([[0], [0], [0]], dtype=torch.float32)
    targets = torch.tensor([[1], [1], [1]], dtype=torch.float32)
    loss = loss_fn(predictions, targets)
    assert loss > 0.9


def test_mcrmse_loss():
    loss_fn = MCRMSELoss(num_targets=2)
    predictions = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    loss = loss_fn(predictions, targets)
    assert loss < 0.1

    predictions = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.float32)
    targets = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float32)
    loss = loss_fn(predictions, targets)
    assert loss > 0.9
