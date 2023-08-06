from pathlib import Path
from commonlit_summaries.data import load_data

DATA_DIR = Path(__file__).parents[3] / "data"


def test_load_train_data_with_summarise():
    data = load_data(DATA_DIR, train=True, summarise=True, checkpoint=None, device="cpu")
    for column in [
        "student_id",
        "prompt_id",
        "prompt_question",
        "prompt_text",
        "reference_summary",
        "text",
        "content",
        "wording",
    ]:
        assert column in data.columns

    assert len(data) > 0
