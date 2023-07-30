from pathlib import Path
from tqdm import tqdm
import torch
from torch.nn import MSELoss
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, get_scheduler

from commonlit_summaries.data import PredictionType
from commonlit_summaries.losses import RMSELoss, MCRMSELoss
from commonlit_summaries.utils import AverageMeter


class Experiment:
    def __init__(
        self,
        prediction_type: PredictionType,
        fold: str,
        model_name: str,
        model_checkpoint: str,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        learning_rate: float,
        train_batch_size: int,
        eval_batch_size: int,
        scheduler: str,
        warmup: float,
        epochs: int,
        accumulation_steps: int,
        loss: str,
        dataloader_workers: int = 2,
        save_dir: Path = Path("./"),
        device: str = "cuda",
    ):
        self.device = device
        self.prediction_type = prediction_type
        self.fold = fold
        self.model_name = model_name
        self.model = self._init_model(model_checkpoint)
        self.loss_fn = self._init_loss_fn(loss)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_num_workers = dataloader_workers
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.train_loss_meter = AverageMeter()
        self.scaler = amp.GradScaler()
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.scheduler = self._init_scheduler(scheduler, warmup)
        self.save_dir = save_dir
        self.step = 1

    def _init_model(self, model_checkpoint: str) -> AutoModelForSequenceClassification:
        num_labels = 2 if self.prediction_type == PredictionType.both else 1
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
        return model.to(self.device)

    def _init_loss_fn(self, name: str) -> torch.nn.Module:
        losses = {"mse": MSELoss, "rmse": RMSELoss, "mcrmse": MCRMSELoss}

        if name not in losses:
            raise ValueError(f"{name} is not a valid loss function.")

        return losses[name]()

    def _init_scheduler(self, scheduler_name: str, warmup_proportion: float) -> LRScheduler:
        epoch_steps = (len(self.train_dataset) // self.train_batch_size) // self.accumulation_steps
        total_steps = epoch_steps * self.epochs
        num_warmup_steps = round(total_steps * warmup_proportion)
        return get_scheduler(scheduler_name, self.optimizer, num_warmup_steps, total_steps)

    def run(self) -> AutoModelForSequenceClassification:
        """Trains with the config specified in the constructor."""
        print(f"Training {self.fold} for {self.epochs} epochs.")
        eval_loss_per_epoch = []

        for epoch in range(1, self.epochs + 1):
            self._train_epoch()
            eval_loss = self._evaluate()
            print(f"Epoch: {epoch} | Eval loss: {eval_loss}")
            self._save(self.fold, epoch)
            eval_loss_per_epoch.append(eval_loss)

        print(f"FOLD {self.fold} SUMMARY: {eval_loss_per_epoch}")
        return self.model, eval_loss_per_epoch

    def _train_epoch(self):
        """Trains the model for one epoch."""
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self.train_loss_meter.reset()
        train_loader = self._get_dataloader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )
        with tqdm(total=len(train_loader), unit="batches") as tepoch:
            for batch in train_loader:
                loss = self._forward_pass(batch, self.train_loss_meter)
                self._backward_pass(loss)
                tepoch.set_postfix({"train_loss": self.train_loss_meter.avg})
                tepoch.update(1)
                self.step += 1

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluates the model on the `eval_dataset` provided and returns the average loss."""
        self.model.eval()
        eval_loader = self._get_dataloader(
            self.eval_dataset, batch_size=self.eval_batch_size, shuffle=False
        )
        eval_loss_meter = AverageMeter()

        with tqdm(total=len(eval_loader), unit="batches") as tepoch:
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self._forward_pass(batch, eval_loss_meter)
                tepoch.set_postfix({"eval_loss": eval_loss_meter.avg})
                tepoch.update(1)

        return eval_loss_meter.avg

    def _forward_pass(
        self, batch: dict[str, torch.Tensor], loss_meter: AverageMeter
    ) -> torch.Tensor:
        """Makes a forward pass with fp16 if cuda is available."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with amp.autocast():
            output = self.model(**batch)
            loss = self.loss_fn(output.logits, batch["labels"])
            loss_meter.update(loss.item(), batch["labels"].shape[0])
            loss = loss / self.accumulation_steps

        return loss

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
        file_name = f"{model_name}-{self.prediction_type.value}-fold-{fold}-epoch-{epoch}.bin"
        save_path = self.save_dir / file_name
        torch.save(self.model.state_dict(), save_path)
