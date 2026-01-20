import re
from collections import Counter
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


_WHITESPACE_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def repetition_penalty(tokens: List[int]):
    if not tokens:
        return 0.0
    data = bytes([t for t in tokens if 0 <= t < 256])
    if not data:
        return 0.0
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
    text = collapse_whitespace(text)
    if not text:
        return 0.0
    run_pen = max(0.0, (max_run_length(text) - 6) / 20.0)
    uniq_pen = 0.0
    if len(text) >= 32:
        uniq_pen = max(0.0, (4 - len(set(text))) / 4.0)
    dom_pen = max(0.0, dominant_char_ratio(text) - 0.18) / 0.5
    return run_pen + uniq_pen + dom_pen


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


def dominant_char_ratio(text: str) -> float:
    letters = [ch.lower() for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    counts = Counter(letters)
    return max(counts.values()) / float(len(letters))


def max_run_length(text: str) -> int:
    max_run = 0
    run = 0
    prev = ""
    for ch in text:
        if ch == prev:
            run += 1
        else:
            prev = ch
            run = 1
        if run > max_run:
            max_run = run
    return max_run


def most_common_bigram_ratio(text: str) -> float:
    if len(text) < 2:
        return 0.0
    counts = Counter(text[i : i + 2] for i in range(len(text) - 1))
    return max(counts.values()) / float(len(text) - 1)


def is_spammy(text: str, min_len_bytes: int = 32) -> bool:
    stripped = text.strip()
    if len(stripped.encode("utf-8")) < min_len_bytes:
        return True
    if not stripped:
        return True
    if dominant_char_ratio(stripped) > 0.35:
        return True
    if max_run_length(stripped) >= 16:
        return True
    if len(stripped) >= 32 and len(set(stripped)) <= 3:
        return True
    if most_common_bigram_ratio(stripped) > 0.25:
        return True
    return False


def quality_override(
    batch_tokens: List[List[int]],
    min_len_bytes: int = 32,
    short_reward: float = -1.0,
    invalid_reward: float = -2.0,
    spam_reward: float = -1.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = []
    values = []
    spam = []
    for tokens in batch_tokens:
        data = bytes([t for t in tokens if 0 <= t < 256])
        if not data:
            mask.append(True)
            values.append(invalid_reward)
            spam.append(False)
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            mask.append(True)
            values.append(invalid_reward)
            spam.append(False)
            continue
        if text == "":
            mask.append(True)
            values.append(invalid_reward)
            spam.append(False)
            continue
        if len(data) < min_len_bytes or only_whitespace_or_punct(text):
            mask.append(True)
            values.append(short_reward)
            spam.append(False)
            continue
        if is_spammy(text, min_len_bytes=min_len_bytes):
            mask.append(True)
            values.append(spam_reward)
            spam.append(True)
            continue
        mask.append(False)
        values.append(0.0)
        spam.append(False)
    return (
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(spam, dtype=torch.bool),
    )


def compute_reward(
    gen_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    batch_tokens: List[List[int]],
    lengths: torch.Tensor,
    max_len: int,
    alpha_len: float,
    beta_rep: float,
    invalid_penalty: float,
    min_len_bytes: int = 32,
    short_reward: float = -1.0,
    invalid_reward: float = -2.0,
    spam_reward: float = -1.5,
):
    cos = cosine_reward(gen_emb, tgt_emb)
    qmask, qvals, spam = quality_override(
        batch_tokens,
        min_len_bytes=min_len_bytes,
        short_reward=short_reward,
        invalid_reward=invalid_reward,
        spam_reward=spam_reward,
    )
    base = torch.where(qmask.to(cos.device), qvals.to(cos.device), cos)
    len_pen = length_penalty(lengths, alpha_len, max_len).to(cos.device)
    rep_raw = batch_repetition_penalty(batch_tokens).to(cos.device)
    rep_pen = (rep_raw * beta_rep).clamp(max=3.0)
    inv_pen = utf8_invalid_penalty(batch_tokens, invalid_penalty).to(cos.device)
    reward = base - len_pen - rep_pen - inv_pen
    return reward, {
        "cosine": cos,
        "len_pen": len_pen,
        "rep_pen": rep_pen,
        "inv_pen": inv_pen,
        "qmask": qmask,
        "qvals": qvals,
        "spam": spam.float(),
    }


def reward(*args, **kwargs):
    return compute_reward(*args, **kwargs)
