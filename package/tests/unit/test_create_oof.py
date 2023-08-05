from pathlib import Path
from tempfile import TemporaryDirectory

from commonlit_summaries.create_oof_df import get_weights_file_path


def test_get_weights_file_path():
    fold = "testfold"

    with TemporaryDirectory() as tempdir:
        weights_dir = Path(tempdir)
        file_path = weights_dir / f"model-{fold}.bin"

        with open(file_path, "w") as f:
            f.write("010101")

        weights_path = get_weights_file_path(fold, weights_dir)

    assert weights_path == file_path
