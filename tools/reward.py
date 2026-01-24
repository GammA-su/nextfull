import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.nn import functional as F

DOM_BYTE_RATIO_THRESH = 0.60
RUN_MAX_THRESH = 16
RUN_MAX_PENALTY_SCALE = 10.0
NONPRINT_RATIO_THRESH = 0.85
NONPRINT_PENALTY_WEIGHT = 2.0
WHITESPACE_BYTES = {0x20, 0x0A, 0x09, 0x0D}
WS_RATIO_THRESH = 0.95
WS_MIN_NONWS = 4


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


def printable_ratio_from_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(1 for b in data if (0x20 <= b <= 0x7E) or b == 0x0A)
    return printable / float(len(data))


def printable_ratio(tokens: List[int], text: str = None) -> float:
    data = bytes([t for t in tokens if 0 <= t < 256])
    if data:
        return printable_ratio_from_bytes(data)
    if text is None:
        return 0.0
    repl = text.encode("utf-8", errors="replace")
    return printable_ratio_from_bytes(repl)


def batch_printable_penalty(
    batch_tokens: List[List[int]],
    min_len_bytes: int,
    threshold: float = NONPRINT_RATIO_THRESH,
    weight: float = NONPRINT_PENALTY_WEIGHT,
):
    ratios = []
    penalties = []
    for tokens in batch_tokens:
        data = bytes([t for t in tokens if 0 <= t < 256])
        ratio = printable_ratio_from_bytes(data)
        ratios.append(ratio)
        if len(data) >= min_len_bytes and ratio < threshold:
            penalties.append((threshold - ratio) * weight)
        else:
            penalties.append(0.0)
    return (
        torch.tensor(ratios, dtype=torch.float32),
        torch.tensor(penalties, dtype=torch.float32),
    )


def dominant_byte_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    return max(counts.values()) / float(len(data))


def max_run_length_bytes(data: bytes) -> int:
    if not data:
        return 0
    max_run = 0
    run = 0
    prev = None
    for b in data:
        if prev is not None and b == prev:
            run += 1
        else:
            prev = b
            run = 1
        if run > max_run:
            max_run = run
    return max_run


def batch_run_penalty(
    batch_tokens: List[List[int]],
    threshold: int = RUN_MAX_THRESH,
    scale: float = RUN_MAX_PENALTY_SCALE,
):
    run_maxes = []
    penalties = []
    for tokens in batch_tokens:
        data = bytes([t for t in tokens if 0 <= t < 256])
        run_max = max_run_length_bytes(data)
        run_maxes.append(run_max)
        if run_max >= threshold:
            penalties.append((run_max - threshold) / float(scale))
        else:
            penalties.append(0.0)
    return (
        torch.tensor(run_maxes, dtype=torch.float32),
        torch.tensor(penalties, dtype=torch.float32),
    )


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
        ratio = printable_ratio(tokens, text=text)
        if len(data) >= min_len_bytes and ratio < NONPRINT_RATIO_THRESH:
            mask.append(True)
            values.append(spam_reward)
            spam.append(True)
            continue
        non_ws_len = sum(1 for b in data if b not in WHITESPACE_BYTES)
        ws_ratio = 1.0 - (non_ws_len / float(len(data))) if data else 1.0
        if len(data) >= min_len_bytes and non_ws_len < max(WS_MIN_NONWS, int(0.05 * len(data))):
            mask.append(True)
            values.append(spam_reward)
            spam.append(True)
            continue
        dom_ratio = dominant_byte_ratio(data)
        if len(data) >= min_len_bytes and dom_ratio >= DOM_BYTE_RATIO_THRESH:
            mask.append(True)
            values.append(spam_reward)
            spam.append(True)
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
    print_ratio, nonprint_pen = batch_printable_penalty(
        batch_tokens, min_len_bytes=min_len_bytes
    )
    print_ratio = print_ratio.to(cos.device)
    nonprint_pen = nonprint_pen.to(cos.device)
    dom_ratio = torch.tensor(
        [dominant_byte_ratio(bytes([t for t in tok if 0 <= t < 256])) for tok in batch_tokens],
        dtype=torch.float32,
        device=cos.device,
    )
    data_lens = torch.tensor(
        [len(bytes([t for t in tok if 0 <= t < 256])) for tok in batch_tokens],
        dtype=torch.float32,
        device=cos.device,
    )
    non_ws_len = torch.tensor(
        [sum(1 for b in bytes([t for t in tok if 0 <= t < 256]) if b not in WHITESPACE_BYTES) for tok in batch_tokens],
        dtype=torch.float32,
        device=cos.device,
    )
    ws_ratio = torch.where(
        data_lens.clamp(min=1.0) > 0,
        1.0 - (non_ws_len / data_lens.clamp(min=1.0)),
        torch.zeros_like(non_ws_len),
    )
    run_max, run_pen = batch_run_penalty(batch_tokens)
    run_max = run_max.to(cos.device)
    run_pen = run_pen.to(cos.device)
    reward = base - len_pen - rep_pen - inv_pen - nonprint_pen - run_pen
    return reward, {
        "cosine": cos,
        "len_pen": len_pen,
        "rep_pen": rep_pen,
        "inv_pen": inv_pen,
        "print_ratio": print_ratio,
        "nonprint_pen": nonprint_pen,
        "dom_ratio": dom_ratio,
        "run_max": run_max,
        "run_pen": run_pen,
        "ws_ratio": ws_ratio,
        "non_ws_len": non_ws_len,
        "qmask": qmask,
        "qvals": qvals,
        "spam": spam.float(),
    }


def reward(*args, **kwargs):
    return compute_reward(*args, **kwargs)
