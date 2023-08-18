import pandas as pd

from commonlit_summaries.metrics import compute_metrics


def test_compute_perfect_mcrmse():
    student_ids = [i for i in range(5)]
    predictions = pd.DataFrame(
        {
            "student_id": student_ids,
            "content": [0.5, 0.3, 0.2, 0.8, 1.0],
            "wording": [0.3, 0.1, 0.75, 0.2, 0.63],
            "predicted_content": [0.5, 0.3, 0.2, 0.8, 1.0],
            "predicted_wording": [0.3, 0.1, 0.75, 0.2, 0.63],
        }
    )
    metrics = compute_metrics(predictions)
    assert metrics["MCRMSE"] == 0.0


def test_compute_mcrmse():
    student_ids = [i for i in range(5)]
    predictions = pd.DataFrame(
        {
            "student_id": student_ids,
            "content": [0.0, 0.0, 0.0, 0.0, 0.0],
            "wording": [0.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_content": [1.0, 1.0, 1.0, 1.0, 1.0],
            "predicted_wording": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    metrics = compute_metrics(predictions)
    assert metrics["MCRMSE"] == 1.0
