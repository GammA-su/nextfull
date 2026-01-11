import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tools.rvq import RVQ
from utils import ensure_dir, setup_runtime


class EmbDataset(Dataset):
    def __init__(self, emb: np.ndarray):
        self.emb = emb

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        return self.emb[idx]


def main(args):
    logger, device = setup_runtime(
        "04_fit_rvq",
        device=args.device,
        threads=args.threads,
        seed=args.seed,
    )
    ensure_dir(args.out_dir)

    emb = np.load(args.emb)
    if emb.shape[1] != args.code_dim:
        raise ValueError(f"emb dim {emb.shape[1]} != code_dim {args.code_dim}")

    ds = EmbDataset(emb.astype(np.float32))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, max(1, args.threads // 2))
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    rvq = RVQ(
        K=args.K,
        V_list=args.V,
        code_dim=args.code_dim,
        ema_decay=args.ema_decay,
        usage_balance_w=args.usage_balance_w,
        device=device,
    ).to(device)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)
            rvq.update(x)
        usage = []
        for k in range(rvq.K):
            counts = rvq.ema_counts[k].detach().cpu().numpy()
            probs = counts / counts.sum()
            entropy = -(probs * np.log(probs + 1e-8)).sum()
            usage.append(entropy)
        logger.info("usage_entropy=%s", [f"{u:.2f}" for u in usage])

    ckpt = {
        "model": rvq.state_dict(),
        "config": {
            "K": args.K,
            "V_list": args.V,
            "code_dim": args.code_dim,
            "ema_decay": args.ema_decay,
            "usage_balance_w": args.usage_balance_w,
        },
    }
    out_path = Path(args.out_dir) / "rvq.pt"
    torch.save(ckpt, str(out_path))
    logger.info("saved=%s", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/sent_emb.npy")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--V", type=int, nargs="+", default=[1024, 4096, 512, 256])
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--ema_decay", type=float, default=0.99)
    ap.add_argument("--usage_balance_w", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()
    if len(args.V) != args.K:
        raise ValueError("V must have K entries")
    main(args)
