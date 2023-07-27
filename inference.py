import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from data import SummaryDataset


class Model:
    def __init__(self, checkpoint: str, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
        self.model = self.model.to("cuda")
        self.model.eval()
        self.collator = DataCollatorWithPadding(
            self.tokenizer, padding="max_length", max_length=max_length, return_tensors="pt"
        )

    def load_weights(self, weights_file: str) -> None:
        print(f"Loading {weights_file}.")
        state_dict = torch.load(weights_file, map_location=torch.device("cuda"))
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(
        self, data: pd.DataFrame, batch_size: int, dataloader_num_workers: int = 2
    ) -> pd.DataFrame:
        dataset = SummaryDataset(self.tokenizer, data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
        predictions = []

        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            for batch in dataloader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                output = self.model(**batch)
                predictions += list(output.logits.squeeze().cpu().numpy())
                tepoch.update(1)

        return predictions
