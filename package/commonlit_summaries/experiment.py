import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch.cuda import amp
from torch.nn import MSELoss, MarginRankingLoss, SmoothL1Loss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, get_scheduler
import wandb

from commonlit_summaries.losses import RMSELoss, MCRMSELoss
from commonlit_summaries.models import (
    CommonlitRegressorModel,
    MeanPooling,
    MaxPooling,
    GeMTextPooling,
    ContextPooling,
)
from commonlit_summaries.utils import AverageMeter


class Experiment:
    def __init__(
        self,
        run_id: str,
        fold: str,
        loss_fn: torch.nn.Module,
        metrics: list[str],
        model_name: str,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        train_batch_size: int,
        eval_batch_size: int,
        epochs: int,
        accumulation_steps: int,
        save_strategy: str,
        log_interval: int,
        dataloader_workers: int = 2,
        save_dir: Path = Path("./"),
        device: str = "cuda",
        use_wandb: bool = True,
    ):
        self.run_id = run_id
        self.device = device
        self.fold = fold
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_num_workers = dataloader_workers
        self.scaler = amp.GradScaler()
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.save_dir = save_dir
        self.save_strategy = save_strategy
        self.step = 1
        self.current_epoch = 1
        self.metrics = metrics
        self.train_loss_meter = {m: AverageMeter() for m in metrics}
        self.log_interval = log_interval
        self.use_wandb = use_wandb

    def run(self) -> tuple[torch.nn.Module, list[float]]:
        """Trains with the config specified in the constructor."""
        print(f"Training {self.fold} for {self.epochs} epochs.")
        eval_metrics = []

        while self.current_epoch <= self.epochs:
            self._train_epoch()

            if self.eval_dataset is not None:
                metrics = self._evaluate()
                eval_metrics.append(metrics)
                print(f"Epoch: {self.current_epoch} | {metrics}")

                if self.use_wandb:
                    log_metrics = {"epoch": self.current_epoch}
                    log_metrics.update({"eval_" + name: metric for name, metric in metrics.items()})
                    wandb.log(log_metrics, step=self.step)

            if self.save_strategy == "all":
                self._save()

            self.current_epoch += 1

        if self.save_strategy == "last":
            self._save()

        print(f"FOLD {self.fold} SUMMARY:")
        print(pd.DataFrame(eval_metrics))
        return self.model, eval_metrics

    def _train_epoch(self) -> None:
        """Trains the model for one epoch."""
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()

        for metric in self.train_loss_meter.values():
            metric.reset()

        train_loader = self._get_dataloader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )

        with tqdm(total=len(train_loader), unit="batches") as tepoch:
            for batch in train_loader:
                loss = self._forward_pass(batch, self.train_loss_meter, push_metrics=self.use_wandb)
                self._backward_pass(loss)
                tepoch.set_postfix({m: self.train_loss_meter[m].avg for m in self.metrics})
                tepoch.update(1)
                self.step += 1

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        """Evaluates the model on the `eval_dataset` provided and returns the average loss."""
        self.model.eval()
        eval_loader = self._get_dataloader(
            self.eval_dataset, batch_size=self.eval_batch_size, shuffle=False
        )
        eval_loss_meter = {m: AverageMeter() for m in self.metrics}

        with tqdm(total=len(eval_loader), unit="batches") as tepoch:
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self._forward_pass(batch, eval_loss_meter, push_metrics=False)
                tepoch.set_postfix({m: eval_loss_meter[m].avg for m in self.metrics})
                tepoch.update(1)

        return {m: eval_loss_meter[m].avg for m in self.metrics}

    def _forward_pass(
        self,
        batch: dict[str, torch.Tensor],
        loss_meter: dict[str, AverageMeter],
        push_metrics: bool = True,
    ) -> torch.Tensor:
        """Makes a forward pass with fp16 if cuda is available."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with amp.autocast():
            output = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            losses = self.loss_fn(output.logits, batch["labels"])
            self._update_metrics(
                loss_meter, losses, batch_size=batch["labels"].shape[0], push_metrics=push_metrics
            )
            loss = losses[0] if isinstance(losses, tuple) else losses
            loss = loss / self.accumulation_steps

        return loss

    def _update_metrics(
        self,
        loss_meters: dict[str, AverageMeter],
        losses: torch.Tensor,
        batch_size,
        push_metrics: bool = True,
    ) -> None:
        # Assign names to metrics
        if len(self.metrics) == 1:
            metrics = {self.metrics[0]: losses}
        else:
            metrics = {m: l for m, l in zip(self.metrics, losses)}

        # Update local metrics
        for metric, loss in metrics.items():
            loss_meters[metric].update(loss.item(), batch_size)

        # Push metrics
        if self.step % self.log_interval == 0 and push_metrics:
            interval_metrics = {"epoch": self.current_epoch}
            interval_metrics.update({m: loss_meters[m].avg for m in self.metrics})
            wandb.log(interval_metrics, step=self.step)

            for metric in self.metrics:
                loss_meters[metric].reset()

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Makes a fp16 backward pass. Only updates the optimizer every `accumulation_steps`."""
        self.scaler.scale(loss).backward()

        if self.step % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

    def _get_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            pin_memory=True,
        )

    def _save(self) -> None:
        model_name = self.model_name.replace("/", "_")
        strategies = {
            "all": f"{self.run_id}-{model_name}-{self.fold}-epoch{self.current_epoch}.bin",
            "last": f"{self.run_id}-{model_name}-{self.fold}.bin",
        }
        file_name = strategies[self.save_strategy]
        save_path = self.save_dir / file_name
        torch.save(self.model.state_dict(), save_path)


class RankingExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_pass(
        self,
        batch: dict[str, torch.Tensor],
        loss_meter: dict[str, AverageMeter],
        push_metrics: bool = True,
    ) -> torch.Tensor:
        return super()._forward_pass(batch, loss_meter, push_metrics)()


def get_model(
    model_checkpoint: str,
    num_labels: int,
    tokenizer_embedding_size: int,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    pooler: str,
    use_attention_head: bool = False,
    device: str = "cuda",
) -> AutoModelForSequenceClassification:
    if pooler == "hf":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
    else:
        pooler_layer = _get_pooling_layer(pooler)
        model = CommonlitRegressorModel(
            model_checkpoint,
            num_labels,
            pooler_layer,
            use_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

    model.resize_token_embeddings(tokenizer_embedding_size)
    return model.to(device)


def _get_pooling_layer(pooler_name: str) -> type[torch.nn.Module]:
    pooling_layers = {
        "context": ContextPooling,  # Hidden state of the first token (CLS)
        "mean": MeanPooling,
        "max": MaxPooling,
        "gemtext": GeMTextPooling,
    }
    if pooler_name not in pooling_layers:
        raise ValueError(f"Unknown pooling layer {pooler_name}.")

    return pooling_layers[pooler_name]


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


def get_optimizer(
    model: AutoModelForSequenceClassification, learning_rate: float, weight_decay: float
) -> Optimizer:
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_lr_scheduler(
    scheduler_name: str, optimizer, warmup_proportion: float, epochs: int, steps_per_epoch: int
) -> LRScheduler:
    total_steps = steps_per_epoch * epochs
    num_warmup_steps = round(total_steps * warmup_proportion)
    return get_scheduler(scheduler_name, optimizer, num_warmup_steps, total_steps)
