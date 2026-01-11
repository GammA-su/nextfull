import numpy as np
import torch
from torch.utils.data import Dataset

BYTE_BOS = 256
BYTE_EOS = 257
BYTE_PAD = 258
BYTE_VOCAB_SIZE = 259


def text_to_bytes(text: str, max_len: int, add_eos: bool = True):
    raw = text.encode("utf-8", errors="ignore")
    ids = list(raw)
    if add_eos:
        if len(ids) < max_len:
            ids.append(BYTE_EOS)
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


def bytes_to_text(byte_ids):
    data = bytes([b for b in byte_ids if 0 <= b < 256])
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def tokens_to_bytes(token_ids):
    return [t for t in token_ids if 0 <= t < 256]


class PlannerSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        return {
            "codes": item["codes"],
            "resid": item["resid"],
            "emb": item["emb"],
        }


class RendererDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "codes": item["codes"],
            "resid": item["resid"],
            "text": item["text"],
            "emb": item["emb"],
        }


def collate_planner(batch, max_steps: int = None):
    lengths = [b["codes"].shape[0] for b in batch]
    if max_steps is not None:
        lengths = [min(l, max_steps) for l in lengths]
    max_len = max(lengths)
    k = batch[0]["codes"].shape[1]
    d_resid = batch[0]["resid"].shape[1]
    d_emb = batch[0]["emb"].shape[1]

    codes = torch.zeros(len(batch), max_len, k, dtype=torch.long)
    resid = torch.zeros(len(batch), max_len, d_resid, dtype=torch.float32)
    emb = torch.zeros(len(batch), max_len, d_emb, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["codes"].shape[0]
        if max_steps is not None and L > max_steps:
            start = L - max_steps
            c = b["codes"][start:]
            r = b["resid"][start:]
            e = b["emb"][start:]
        else:
            c = b["codes"]
            r = b["resid"]
            e = b["emb"]
        L = c.shape[0]
        codes[i, :L] = c
        resid[i, :L] = r
        emb[i, :L] = e
    return codes, resid, emb, torch.tensor(lengths, dtype=torch.long)


def collate_renderer(batch, max_len: int):
    codes = torch.stack([b["codes"] for b in batch]).long()
    resid = torch.stack([b["resid"] for b in batch]).float()
    emb = torch.stack([b["emb"] for b in batch]).float()

    byte_ids = []
    lengths = []
    for b in batch:
        ids = text_to_bytes(b["text"], max_len=max_len, add_eos=True)
        lengths.append(len(ids))
        if len(ids) < max_len:
            ids = ids + [BYTE_PAD] * (max_len - len(ids))
        byte_ids.append(ids)

    byte_ids = torch.tensor(byte_ids, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return codes, resid, emb, byte_ids, lengths
