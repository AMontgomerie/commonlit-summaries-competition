from transformers import AutoTokenizer, PreTrainedTokenizerBase


SPECIAL_TOKENS = {
    "title": "<title>",
    "text": "<text>",
    "question": "<question>",
    "summary": "<summary>",
}


def setup_tokenizer(checkpoint: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.keys())})
    return tokenizer
