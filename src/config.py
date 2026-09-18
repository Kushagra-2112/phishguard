from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """A dict that also supports attribute access, recursively.
    So cfg["model"]["text_max_len"] can also be written cfg.model.text_max_len
    """
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        self[name] = value


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def resolve(relative: str | Path) -> Path:
    """Turn a path from the config (e.g. 'data/raw') into an absolute path,
    regardless of which folder a script is run from."""
    relative = Path(relative)
    return relative if relative.is_absolute() else PROJECT_ROOT / relative


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")