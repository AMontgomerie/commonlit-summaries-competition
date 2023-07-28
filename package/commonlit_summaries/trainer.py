import os
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

from commonlit_summaries.data import PredictionType
from commonlit_summaries.utils import AverageMeter


class Trainer:
    def __init__(
        self,
        prediction_type: PredictionType,
        fold: str,
        model_checkpoint: str,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        learning_rate: float,
        train_batch_size: int,
        eval_batch_size: int,
        device: str = "cuda",
    ):
        self.device = device
        self.prediction_type = prediction_type
        self.fold = fold
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=1
        )
        self.model_checkpoint = model_checkpoint
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_num_workers = 2
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.train_loss = AverageMeter()
        self.scaler = amp.GradScaler()
        self.save_dir = "./"

    def train(self, epochs: int) -> AutoModelForSequenceClassification:
        rmse_per_epoch = []

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self._train_epoch()
            mse = self._evaluate()
            rmse = np.sqrt(mse)
            print(f"Epoch: {epoch} | MSE: {mse} | RMSE: {rmse}")
            self._save(self.fold, epoch)
            rmse_per_epoch.append(rmse)

        print(f"EVAL FOLD {self.fold} SUMMARY:")
        print(rmse_per_epoch)

    def _train_epoch(self):
        self.model.train()
        self.train_loss.reset()
        train_loader = self._get_dataloader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )
        with tqdm(total=len(train_loader), unit="batches") as tepoch:
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._model_fn(batch)
                self.train_loss.update(loss.item(), self.train_batch_size)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                tepoch.set_postfix({"train_loss": self.train_loss.avg})
                tepoch.update(1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        eval_loader = self._get_dataloader(
            self.eval_dataset, batch_size=self.eval_batch_size, shuffle=False
        )
        eval_loss = AverageMeter()

        with tqdm(total=len(eval_loader), unit="batches") as tepoch:
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self._model_fn(batch)
                eval_loss.update(loss.item(), self.eval_batch_size)
                tepoch.set_postfix({"eval_loss": eval_loss.avg})
                tepoch.update(1)

        return eval_loss.avg

    def _model_fn(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with amp.autocast():
            output = self.model(**batch)

        return output.loss

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
        model_name = self.model_checkpoint.replace("/", "_")
        file_name = f"{model_name}-{self.prediction_type.value}-fold-{fold}-epoch-{epoch}.bin"
        save_path = os.path.join(self.save_dir, file_name)
        torch.save(self.model.state_dict(), save_path)
