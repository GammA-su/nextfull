import json
import logging
import os
import random
from pathlib import Path

import numpy as np

try:
    import regex as re
except Exception:  # pragma: no cover - optional fallback
    import re

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, level: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    if level is None:
        level = os.environ.get("SENTCODELM_LOG_LEVEL", "INFO")
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def configure_threads(requested: int = 16) -> int:
    if requested <= 0:
        return 0
    cpu_count = os.cpu_count() or requested
    threads = min(requested, cpu_count)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, threads // 2))
    except Exception:
        pass
    return threads


def detect_faiss():
    try:
        import faiss
    except Exception:
        return False, 0
    try:
        gpu_count = int(faiss.get_num_gpus())
    except Exception:
        gpu_count = 0
    return True, gpu_count


def select_device(device: str = None, logger: logging.Logger = None, deterministic: bool = False):
    try:
        import torch
    except Exception as exc:
        if logger:
            logger.warning("torch unavailable: %s", exc)
        return None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        if logger:
            logger.warning("cuda requested but not available; falling back to cpu")
        device = "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not deterministic:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = False
        except Exception:
            pass
    return torch_device


def setup_runtime(
    name: str,
    device: str = None,
    threads: int = 16,
    seed: int = None,
    deterministic: bool = False,
):
    logger = get_logger(name)
    thread_count = configure_threads(threads)
    if seed is not None:
        set_seed(seed, deterministic=deterministic)
    torch_device = select_device(device, logger, deterministic=deterministic)
    faiss_available, faiss_gpu_count = detect_faiss()

    gpu_enabled = False
    gpu_available = False
    gpu_count = 0
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_available = False
        gpu_count = 0
    if torch_device is not None and getattr(torch_device, "type", None) == "cuda":
        gpu_enabled = True

    device_name = torch_device.type if torch_device is not None else "cpu"
    logger.info(
        "runtime: device=%s gpu_enabled=%s gpu_available=%s gpus=%d threads=%d",
        device_name,
        gpu_enabled,
        gpu_available,
        gpu_count,
        thread_count,
    )
    logger.info(
        "faiss: available=%s gpu=%s gpus=%d",
        faiss_available,
        faiss_gpu_count > 0,
        faiss_gpu_count,
    )
    if not gpu_enabled:
        logger.info("cpu: threads=%d", thread_count)
    return logger, torch_device


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
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
