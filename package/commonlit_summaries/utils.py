import numpy as np
import os
from pathlib import Path
import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_weights_file_path(fold: str, weights_dir: Path) -> Path:
    for filename in weights_dir.iterdir():
        if fold in str(filename) and filename.suffix == ".bin":
            return filename.absolute()

    raise FileNotFoundError(f"Couldn't find a weights file for fold {fold}.")
