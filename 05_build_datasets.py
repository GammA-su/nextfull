import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from utils import ensure_dir, read_jsonl, setup_runtime


def main(args):
    logger, _ = setup_runtime(
        "05_build_datasets",
        threads=args.threads,
        seed=args.seed,
    )
    ensure_dir(args.out_dir)

    sentences = read_jsonl(str(Path(args.data_dir) / "sentences.jsonl"))
    sequences = read_jsonl(str(Path(args.data_dir) / "sequences.jsonl"))
    sentences = sorted(sentences, key=lambda x: x["sid"])

    codes = np.load(str(Path(args.data_dir) / "sent_codes.npy"))
    resid = np.load(str(Path(args.data_dir) / "sent_resid.npy"))
    emb = np.load(str(Path(args.data_dir) / "sent_emb.npy"))

    if codes.shape[0] != len(sentences):
        raise ValueError("codes size mismatch")
    if resid.shape[0] != len(sentences):
        raise ValueError("resid size mismatch")
    if emb.shape[0] != len(sentences):
        raise ValueError("emb size mismatch")

    sent_text = [s["text"] for s in sentences]

    doc_ids = [seq["doc_id"] for seq in sequences]
    doc_ids = list(set(doc_ids))
    random.shuffle(doc_ids)
    n_val = max(1, int(len(doc_ids) * args.val_frac))
    val_docs = set(doc_ids[:n_val])

    train = {"planner": [], "renderer": []}
    val = {"planner": [], "renderer": []}

    total = len(sequences)
    logger.info("sequences=%d val_frac=%.3f", total, args.val_frac)
    start = time.time()
    last_log = start
    for idx, seq in enumerate(sequences, start=1):
        sids = seq["sids"]
        if len(sids) < 2:
            continue
        is_val = seq["doc_id"] in val_docs
        pack = val if is_val else train

        codes_seq = torch.tensor(codes[sids], dtype=torch.long)
        resid_seq = torch.tensor(resid[sids], dtype=torch.float32)
        emb_seq = torch.tensor(emb[sids], dtype=torch.float32)
        pack["planner"].append({"codes": codes_seq, "resid": resid_seq, "emb": emb_seq})

        for sid in sids:
            pack["renderer"].append(
                {
                    "codes": torch.tensor(codes[sid], dtype=torch.long),
                    "resid": torch.tensor(resid[sid], dtype=torch.float32),
                    "emb": torch.tensor(emb[sid], dtype=torch.float32),
                    "text": sent_text[sid],
                }
            )
        if args.log_every > 0 and idx % args.log_every == 0:
            now = time.time()
            step = now - last_log
            rate = (args.log_every / step) if step > 0 else 0.0
            pct = 100.0 * idx / total if total else 0.0
            logger.info(
                "progress: %d/%d (%.2f%%) rate=%.1f seq/s",
                idx,
                total,
                pct,
                rate,
            )
            last_log = now

    torch.save(train, str(Path(args.out_dir) / "packed_train.pt"))
    torch.save(val, str(Path(args.out_dir) / "packed_val.pt"))
    logger.info(
        "train_planner=%d train_renderer=%d",
        len(train["planner"]),
        len(train["renderer"]),
    )
    logger.info(
        "val_planner=%d val_renderer=%d",
        len(val["planner"]),
        len(val["renderer"]),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    main(args)
