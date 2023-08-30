import torch

from commonlit_summaries.losses import RMSELoss, MCRMSELoss, TiedMarginRankingLoss


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
    mcrmse, rmse1, rmse2 = loss_fn(predictions, targets)
    assert mcrmse < 0.1

    predictions = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.float32)
    targets = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float32)
    mcrmse, rmse1, rmse2 = loss_fn(predictions, targets)
    assert mcrmse > 0.9


def test_tied_margin_ranking_loss():
    loss_fn = TiedMarginRankingLoss(margin=0.5)

    # Test that a correct prediction of a tie returns 0 loss
    predictions1 = torch.tensor([0.1])
    predictions2 = torch.tensor([0.1])
    targets = torch.tensor([0])
    loss = loss_fn(predictions1, predictions2, targets)
    assert loss.item() == 0

    # Test that a false prediction of a tie returns the margin
    targets = torch.tensor([1])
    loss = loss_fn(predictions1, predictions2, targets)
    assert loss.item() == 0.5

    # Test that an incorrect prediction of input1 > input2 is penalised
    predictions1 = torch.tensor([1])
    predictions2 = torch.tensor([0])
    targets = torch.tensor([-1])
    loss = loss_fn(predictions1, predictions2, targets)
    assert loss.item() > 0.5

    # Test that a correct prediction of input1 > input2 returns 0
    targets = torch.tensor([1])
    loss = loss_fn(predictions1, predictions2, targets)
    assert loss.item() == 0

    # Test that failing to detect a tie is penalised
    targets = torch.tensor([0])
    loss = loss_fn(predictions1, predictions2, targets)
    assert loss.item() > 0.5
