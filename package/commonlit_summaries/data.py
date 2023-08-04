from enum import Enum
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer


class PromptType(Enum):
    summary = "summary"
    title = "title"
    question = "question"
    text = "text"


class PredictionType(Enum):
    content = "content"
    wording = "wording"
    both = "both"


class SummaryDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data: pd.DataFrame,
        prompt_types: list[PromptType],
        prediction_type: PredictionType | None = None,
        fix_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.prompt_types = prompt_types
        self.prediction_type = prediction_type
        self.fix_length = fix_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.data.loc[index]
        sep_token = self.tokenizer.sep_token or "\n\n"
        prompts = {
            PromptType.summary: sample.text,
            PromptType.title: sample.prompt_title,
            PromptType.question: sample.prompt_question,
            PromptType.text: sample.prompt_text,
        }
        texts = [prompts[prompt_type] for prompt_type in self.prompt_types]
        text = sep_token.join(texts)

        # If we're not using a data collator then we can do truncation, padding, and conversion to
        # tensors here.
        if self.fix_length:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.fix_length,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.squeeze(dim=0) for k, v in inputs.items()}

        # Otherwise just encode the sequence and leave the rest to the data collator.
        else:
            inputs = self.tokenizer(text)

        # Determine which targets to use.
        if self.prediction_type:
            label_data = {
                PredictionType.content: sample.content,
                PredictionType.wording: sample.wording,
                PredictionType.both: [sample.content, sample.wording],
            }
            label = label_data[self.prediction_type]
            inputs["labels"] = torch.tensor(label, dtype=torch.float32)

        return inputs


def load_data(data_dir: Path, train: bool = True):
    split = "train" if train else "test"
    prompts = pd.read_csv(data_dir / f"prompts_{split}.csv")
    summaries = pd.read_csv(data_dir / f"summaries_{split}.csv")
    data = summaries.merge(prompts, on="prompt_id")
    columns = ["student_id", "prompt_id", "prompt_title", "prompt_question", "prompt_text", "text"]

    if train:
        columns += ["content", "wording"]

    data = data.loc[:, columns]
    return data
