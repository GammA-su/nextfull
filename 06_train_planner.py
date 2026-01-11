import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.data import PlannerSequenceDataset, collate_planner
from tools.planner import Planner
from tools.rvq import load_rvq
from utils import ensure_dir, setup_runtime


def masked_ce(logits, targets, mask):
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = loss * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return loss.sum() / denom


def info_nce(pred, target, tau: float):
    sim = pred @ target.t() / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def main(args):
    logger, device = setup_runtime(
        "06_train_planner",
        device=args.device,
        threads=args.threads,
        seed=args.seed,
    )
    ensure_dir(args.out_dir)

    rvq = load_rvq(args.rvq, device=device)

    train_pack = torch.load(args.train_data, map_location="cpu")
    val_pack = torch.load(args.val_data, map_location="cpu")

    train_ds = PlannerSequenceDataset(train_pack["planner"])
    val_ds = PlannerSequenceDataset(val_pack["planner"])

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, max(1, args.threads // 2))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_planner(b, max_steps=args.max_steps),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_planner(b, max_steps=args.max_steps),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    d_resid = train_pack["planner"][0]["resid"].shape[1]
    model = Planner(
        K=rvq.K,
        V_list=rvq.V_list,
        d_resid=d_resid,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_steps=args.max_steps,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for codes, resid, emb, lengths in pbar:
            codes = codes.to(device, non_blocking=True)
            resid = resid.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            code_logits, pred_resid, _ = model(codes, resid, lengths=lengths)
            mask = (
                torch.arange(codes.size(1) - 1, device=device).unsqueeze(0)
                < (lengths - 1).unsqueeze(1)
            )

            code_loss = 0.0
            for k in range(rvq.K):
                logits_k = code_logits[k][:, :-1, :].reshape(-1, rvq.V_list[k])
                targets_k = codes[:, 1:, k].reshape(-1)
                mask_k = mask.reshape(-1)
                code_loss = code_loss + masked_ce(logits_k, targets_k, mask_k)

            pred = pred_resid[:, :-1, :].reshape(-1, pred_resid.size(-1))
            tgt = emb[:, 1:, :].reshape(-1, emb.size(-1))
            mask_flat = mask.reshape(-1)
            pred = pred[mask_flat]
            tgt = tgt[mask_flat]

            if pred.size(0) > 1:
                nce_loss = info_nce(pred, tgt, args.tau)
            else:
                nce_loss = torch.tensor(0.0, device=device)

            loss = code_loss + args.lambda_nce * nce_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            pbar.set_postfix(loss=float(loss), code=float(code_loss), nce=float(nce_loss))

        if len(val_ds) > 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for codes, resid, emb, lengths in val_loader:
                    codes = codes.to(device, non_blocking=True)
                    resid = resid.to(device, non_blocking=True)
                    emb = emb.to(device, non_blocking=True)
                    lengths = lengths.to(device, non_blocking=True)
                    code_logits, pred_resid, _ = model(codes, resid, lengths=lengths)
                    mask = (
                        torch.arange(codes.size(1) - 1, device=device).unsqueeze(0)
                        < (lengths - 1).unsqueeze(1)
                    )
                    code_loss = 0.0
                    for k in range(rvq.K):
                        logits_k = code_logits[k][:, :-1, :].reshape(-1, rvq.V_list[k])
                        targets_k = codes[:, 1:, k].reshape(-1)
                        mask_k = mask.reshape(-1)
                        code_loss = code_loss + masked_ce(logits_k, targets_k, mask_k)
                    pred = pred_resid[:, :-1, :].reshape(-1, pred_resid.size(-1))
                    tgt = emb[:, 1:, :].reshape(-1, emb.size(-1))
                    mask_flat = mask.reshape(-1)
                    pred = pred[mask_flat]
                    tgt = tgt[mask_flat]
                    if pred.size(0) > 1:
                        nce_loss = info_nce(pred, tgt, args.tau)
                    else:
                        nce_loss = torch.tensor(0.0, device=device)
                    losses.append(float(code_loss + args.lambda_nce * nce_loss))
            logger.info("val_loss=%.4f", sum(losses) / max(1, len(losses)))

    ckpt = {
        "model": model.state_dict(),
        "config": {
            "K": rvq.K,
            "V_list": rvq.V_list,
            "d_resid": d_resid,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "max_steps": args.max_steps,
            "dropout": args.dropout,
        },
    }
    out_path = Path(args.out_dir) / "planner.pt"
    torch.save(ckpt, str(out_path))
    logger.info("saved=%s", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", default="data/packed_train.pt")
    ap.add_argument("--val_data", default="data/packed_val.pt")
    ap.add_argument("--rvq", default="out/rvq.pt")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--max_steps", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()
    main(args)
