from enum import Enum
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer


class PredictionType(Enum):
    content = "content"
    wording = "wording"
    both = "both"


class SummaryDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data: pd.DataFrame,
        prediction_type: PredictionType | None = None,
        fix_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.type = prediction_type
        self.fix_length = fix_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.data.loc[index]

        # If we're not using a data collator then we can do truncation, padding, and conversion to
        # tensors here.
        if self.fix_length:
            inputs = self.tokenizer(
                sample.prompt_question,
                sample.text,
                truncation=True,
                max_length=self.fix_length,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.squeeze(dim=0) for k, v in inputs.items()}

        # Otherwise just encode the sequence and leave the rest to the data collator.
        else:
            inputs = self.tokenizer(sample.prompt_question, sample.text)

        if self.type:
            label_data = {
                PredictionType.content: sample.content,
                PredictionType.wording: sample.wording,
                PredictionType.both: [sample.content, sample.wording],
            }
            label = label_data[self.type]
            inputs["labels"] = torch.tensor(label, dtype=torch.float32)

        return inputs


def load_data(data_dir: Path, train: bool = True):
    split = "train" if train else "test"
    prompts = pd.read_csv(data_dir / f"prompts_{split}.csv")
    summaries = pd.read_csv(data_dir / f"summaries_{split}.csv")
    data = summaries.merge(prompts, on="prompt_id")
    columns = ["student_id", "prompt_id", "prompt_question", "text"]

    if train:
        columns += ["content", "wording"]

    data = data.loc[:, columns]
    return data
