import argparse
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tools.data import BYTE_PAD, text_to_bytes
from tools.encoder import ByteEncoder
from utils import ensure_dir, read_jsonl, setup_runtime, simple_train_val_split


class PairDataset(Dataset):
    def __init__(self, pairs, max_len: int):
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ctx, nxt = self.pairs[idx]
        return ctx, nxt


def collate_pairs(batch, max_len: int):
    ctx_ids = []
    nxt_ids = []
    for ctx, nxt in batch:
        ctx_b = text_to_bytes(ctx, max_len=max_len, add_eos=True)
        nxt_b = text_to_bytes(nxt, max_len=max_len, add_eos=True)
        if len(ctx_b) < max_len:
            ctx_b = ctx_b + [BYTE_PAD] * (max_len - len(ctx_b))
        if len(nxt_b) < max_len:
            nxt_b = nxt_b + [BYTE_PAD] * (max_len - len(nxt_b))
        ctx_ids.append(ctx_b)
        nxt_ids.append(nxt_b)
    ctx_ids = torch.tensor(ctx_ids, dtype=torch.long)
    nxt_ids = torch.tensor(nxt_ids, dtype=torch.long)
    return ctx_ids, nxt_ids


def build_pairs(sentences, sequences, window: int):
    sent_map = {s["sid"]: s["text"] for s in sentences}
    pairs = []
    for seq in sequences:
        sids = seq["sids"]
        for i in range(len(sids) - 1):
            start = max(0, i - window + 1)
            ctx = " ".join(sent_map[sid] for sid in sids[start : i + 1])
            nxt = sent_map[sids[i + 1]]
            pairs.append((ctx, nxt))
    return pairs


def info_nce_loss(ctx_emb, tgt_emb, tau: float):
    logits = ctx_emb @ tgt_emb.t() / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def main(args):
    logger, device = setup_runtime(
        "02_train_sentence_encoder",
        device=args.device,
        threads=args.threads,
        seed=args.seed,
    )
    ensure_dir(args.out_dir)

    sentences = read_jsonl(str(Path(args.data_dir) / "sentences.jsonl"))
    sequences = read_jsonl(str(Path(args.data_dir) / "sequences.jsonl"))

    pairs = build_pairs(sentences, sequences, window=args.context_window)
    train_pairs, val_pairs = simple_train_val_split(pairs, args.val_frac, args.seed)
    logger.info(
        "pairs=%d train=%d val=%d",
        len(pairs),
        len(train_pairs),
        len(val_pairs),
    )

    train_ds = PairDataset(train_pairs, args.max_bytes)
    val_ds = PairDataset(val_pairs, args.max_bytes)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, max(1, args.threads // 2))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pairs(b, args.max_bytes),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pairs(b, args.max_bytes),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    model = ByteEncoder(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_bytes,
        d_emb=args.d_emb,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for ctx_ids, nxt_ids in pbar:
            ctx_ids = ctx_ids.to(device, non_blocking=True)
            nxt_ids = nxt_ids.to(device, non_blocking=True)
            ctx_emb = model(ctx_ids)
            nxt_emb = model(nxt_ids)
            loss = info_nce_loss(ctx_emb, nxt_emb, args.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            pbar.set_postfix(loss=float(loss.detach()))

        if val_pairs:
            model.eval()
            losses = []
            with torch.no_grad():
                for ctx_ids, nxt_ids in val_loader:
                    ctx_ids = ctx_ids.to(device, non_blocking=True)
                    nxt_ids = nxt_ids.to(device, non_blocking=True)
                    ctx_emb = model(ctx_ids)
                    nxt_emb = model(nxt_ids)
                    vloss = info_nce_loss(ctx_emb, nxt_emb, args.tau)
                    losses.append(float(vloss.detach()))
            logger.info("val_loss=%.4f", sum(losses) / max(1, len(losses)))
            model.train()

    ckpt = {
        "model": model.state_dict(),
        "config": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "max_len": args.max_bytes,
            "d_emb": args.d_emb,
            "dropout": args.dropout,
        },
    }
    torch.save(ckpt, str(Path(args.out_dir) / "enc.pt"))
    logger.info("saved=%s", Path(args.out_dir) / "enc.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--max_bytes", type=int, default=256)
    ap.add_argument("--context_window", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_emb", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()
    main(args)
