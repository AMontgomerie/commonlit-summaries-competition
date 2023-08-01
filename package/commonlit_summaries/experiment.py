import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch.cuda import amp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification
import wandb

from commonlit_summaries.utils import AverageMeter


class Experiment:
    def __init__(
        self,
        fold: str,
        loss_fn: torch.nn.Module,
        metrics: list[str],
        model_name: str,
        model: AutoModelForSequenceClassification,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataset: Dataset,
        eval_dataset: Dataset,
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
        self.metrics = metrics
        self.train_loss_meter = {m: AverageMeter() for m in metrics}
        self.log_interval = log_interval
        self.use_wandb = use_wandb

    def run(self) -> tuple[AutoModelForSequenceClassification, list[float]]:
        """Trains with the config specified in the constructor."""
        print(f"Training {self.fold} for {self.epochs} epochs.")
        eval_metrics = []

        for epoch in range(1, self.epochs + 1):
            self._train_epoch()
            metrics = self._evaluate()

            if self.use_wandb:
                wandb.log({"eval_" + name: metric for name, metric in metrics.items()}, step=epoch)

            print(f"Epoch: {epoch} | {metrics}")

            if self.save_strategy == "all":
                self._save(self.fold, epoch)

            eval_metrics.append(metrics)

        if self.save_strategy == "last":
            self._save(self.fold, epoch)

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
    def _evaluate(self) -> float:
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
            output = self.model(**batch)
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
            wandb.log(metrics, step=self.step)

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

    def _save(self, fold: str, epoch: int) -> None:
        model_name = self.model_name.replace("/", "_")
        file_name = f"{model_name}-fold-{fold}-epoch-{epoch}.bin"
        save_path = self.save_dir / file_name
        torch.save(self.model.state_dict(), save_path)
