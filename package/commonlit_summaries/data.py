from enum import Enum
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

from commonlit_summaries.tokenizer import SPECIAL_TOKENS


class PromptType(Enum):
    summary = "summary"
    title = "title"
    question = "question"
    text = "text"
    reference_summary = "reference_summary"


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
        seed: int = None,  # unused
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
        prompts = {
            PromptType.summary: SPECIAL_TOKENS["summary"] + " " + sample.text,
            PromptType.title: SPECIAL_TOKENS["title"] + " " + sample.prompt_title,
            PromptType.question: SPECIAL_TOKENS["question"] + " " + sample.prompt_question,
            PromptType.text: SPECIAL_TOKENS["text"] + " " + sample.prompt_text,
            PromptType.reference_summary: SPECIAL_TOKENS["reference_summary"]
            + " "
            + sample.reference_summary,
        }
        texts = [prompts[prompt_type] for prompt_type in self.prompt_types]
        text = " ".join(texts)
        inputs = self._tokenize(text)

        # Determine which targets to use.
        if self.prediction_type:
            label_data = {
                PredictionType.content: [sample.content],
                PredictionType.wording: [sample.wording],
                PredictionType.both: [sample.content, sample.wording],
            }
            label = label_data[self.prediction_type]
            inputs["labels"] = torch.tensor(label, dtype=torch.float32)

        return inputs

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        # If we're not using a data collator then we can do truncation, padding, and conversion to
        # tensors here.
        if self.fix_length:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.fix_length,
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=False,
            )
            inputs = {k: v.squeeze(dim=0) for k, v in inputs.items()}

        # Otherwise just encode the sequence and leave the rest to the data collator.
        else:
            inputs = self.tokenizer(text)

        return inputs


class SummaryRankingDataset(SummaryDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data: pd.DataFrame,
        prompt_types: list[PromptType],
        prediction_type: PredictionType | None = None,
        fix_length: int | None = None,
        seed: int = 666,
    ):
        super().__init__(tokenizer, data, prompt_types, prediction_type, fix_length)
        self.index_pair = list(data.sample(frac=1.0, random_state=seed).index)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        paired_index = self.index_pair[index]
        input1 = super().__getitem__(index)
        input2 = super().__getitem__(paired_index)

        # Label is 0 if the two targets are the same, 1 if the first target is greater, and -1 if
        # the second target is greater.
        labels = torch.where(
            input1["labels"] == input2["labels"],
            0,
            torch.where(input1["labels"] > input2["labels"], 1, -1),
        )
        return input1, input2, labels


def load_data(data_dir: Path, train: bool = True, summarise: bool = False, **summarizer_kwargs):
    split = "train" if train else "test"
    prompts = pd.read_csv(data_dir / f"prompts_{split}.csv")
    summaries = pd.read_csv(data_dir / f"summaries_{split}.csv")

    prompts["reference_summary"] = ""

    if summarise:
        prompt_ids = list(prompts.prompt_id)
        full_texts = list(prompts.prompt_text)
        prompts["reference_summary"] = generate_summaries(
            prompt_ids, full_texts, **summarizer_kwargs
        )

    data = summaries.merge(prompts, on="prompt_id")
    columns = [
        "student_id",
        "prompt_id",
        "prompt_title",
        "prompt_question",
        "prompt_text",
        "text",
        "reference_summary",
    ]

    if train:
        columns += ["content", "wording"]

    data = data.loc[:, columns]
    return data


def generate_summaries(
    prompt_ids: list[str],
    texts: list[str],
    checkpoint: str = "facebook/bart-large-cnn",
    max_length: int = 1024,
    min_length: int = 56,
    model_max_length: int = 1024,
    device: str = "cuda",
) -> pd.DataFrame:
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, max_length=model_max_length)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=model_max_length)
    summarizer = pipeline(
        "summarization", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1
    )
    summaries = []

    for id, text in zip(prompt_ids, texts):
        # Don't summarise if the text is already short.
        tokenized_text = tokenizer(text, return_attention_mask=False)

        if isinstance(tokenized_text, dict):
            tokenized_text = tokenized_text["input_ids"]

        if len(tokenized_text) < max_length:
            print(f"Prompt {id} is shorter than max summary length {max_length}. Using full text.")
            summaries.append(text)

        # If the text is long then shorten it within the specified range.
        else:
            print(f"Prompt {id} has length {len(tokenized_text)}. Generating summary.")
            summary = summarizer(
                text, truncation=True, min_length=min_length, max_length=max_length
            )
            summaries.append(summary["summary_text"])

    return summaries
