from transformers import AutoTokenizer, AutoModelForSequenceClassification

from commonlit_summaries.models import CommonlitRegressorModel, MeanPooling


def test_model_output_shape():
    """Tests that a model with a custom head outputs the same shaped tensor as a huggingface model
    with a SequenceClassification head and the same number of labels
    """
    checkpoint = "distilroberta-base"
    num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pooler = MeanPooling()
    model = CommonlitRegressorModel(checkpoint, num_labels=num_labels, pooler=pooler)
    hf_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    text = "This is some text used as an input to the model."
    inputs = tokenizer(text, return_tensors="pt")
    output = model(**inputs)
    hf_output = hf_model(**inputs)

    assert output.logits.shape == hf_output.logits.shape


def test_model_output_shape_with_attention_head():
    checkpoint = "distilroberta-base"
    num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pooler = MeanPooling()
    model = CommonlitRegressorModel(
        checkpoint, num_labels=num_labels, use_attention_head=True, pooler=pooler
    )
    hf_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    text = "This is some text used as an input to the model."
    inputs = tokenizer(text, return_tensors="pt")
    output = model(**inputs)
    hf_output = hf_model(**inputs)

    assert output.logits.shape == hf_output.logits.shape
