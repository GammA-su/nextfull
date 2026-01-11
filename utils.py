import json
import os
import random
from pathlib import Path

import numpy as np

try:
    import regex as re
except Exception:  # pragma: no cover - optional fallback
    import re

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def simple_train_val_split(items, val_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(len(items) * val_frac))
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, item in enumerate(items):
        if i in val_idx:
            val.append(item)
        else:
            train.append(item)
    return train, val
