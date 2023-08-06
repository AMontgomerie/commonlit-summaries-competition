from commonlit_summaries.tokenizer import setup_tokenizer, SPECIAL_TOKENS


def test_add_special_tokens():
    tokenizer = setup_tokenizer("distilroberta-base")

    for token in SPECIAL_TOKENS:
        assert token in tokenizer.special_tokens_map["additional_special_tokens"]
