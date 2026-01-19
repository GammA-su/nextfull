import math
from typing import List, Tuple

import torch
from torch.nn import functional as F


def cosine_reward(gen_emb: torch.Tensor, tgt_emb: torch.Tensor):
    gen = F.normalize(gen_emb, dim=-1)
    tgt = F.normalize(tgt_emb, dim=-1)
    return (gen * tgt).sum(dim=-1)


def length_penalty(lengths: torch.Tensor, alpha_len: float, max_len: int):
    if alpha_len <= 0:
        return torch.zeros_like(lengths, dtype=torch.float32)
    return alpha_len * (lengths.float() / float(max_len))


def repetition_penalty(tokens: List[int]):
    if not tokens:
        return 0.0
    uniq = len(set(tokens))
    return float(len(tokens) - uniq) / float(len(tokens))


def batch_repetition_penalty(batch_tokens: List[List[int]]):
    return torch.tensor([repetition_penalty(t) for t in batch_tokens], dtype=torch.float32)


def utf8_invalid_penalty(batch_tokens: List[List[int]], penalty: float):
    out = []
    for tokens in batch_tokens:
        data = bytes([t for t in tokens if 0 <= t < 256])
        try:
            data.decode("utf-8")
            out.append(0.0)
        except UnicodeDecodeError:
            out.append(penalty)
    return torch.tensor(out, dtype=torch.float32)


def only_whitespace_or_punct(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return not any(ch.isalnum() for ch in stripped)


def quality_override(
    batch_tokens: List[List[int]],
    min_len_bytes: int = 32,
    short_reward: float = -1.0,
    invalid_reward: float = -2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = []
    values = []
    for tokens in batch_tokens:
        data = bytes([t for t in tokens if 0 <= t < 256])
        if not data:
            mask.append(True)
            values.append(invalid_reward)
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            mask.append(True)
            values.append(invalid_reward)
            continue
        if text == "":
            mask.append(True)
            values.append(invalid_reward)
            continue
        if len(data) < min_len_bytes or only_whitespace_or_punct(text):
            mask.append(True)
            values.append(short_reward)
            continue
        mask.append(False)
        values.append(0.0)
    return torch.tensor(mask, dtype=torch.bool), torch.tensor(values, dtype=torch.float32)
