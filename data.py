from enum import Enum
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer


class PredictionType(Enum):
    content = "Content"
    wording = "Wording"


class SummaryDataset:
    def __init__(
        self, tokenizer: AutoTokenizer, data: pd.DataFrame, prediction_type: PredictionType
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.type = prediction_type

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.data.loc[index]
        inputs = self.tokenizer(sample.prompt_question, sample.text, truncation=True)
        label = sample.content if self.type == PredictionType.content else sample.wording
        inputs["labels"] = torch.tensor(label, dtype=torch.float32)
        return inputs


def load_data(data_dir: Path):
    prompts = pd.read_csv(data_dir / "prompts_train.csv")
    summaries = pd.read_csv(data_dir / "summaries_train.csv")
    data = summaries.merge(prompts, on="prompt_id")
    data = data.loc[:, ["prompt_id", "prompt_question", "text", "content", "wording"]]
    return data
