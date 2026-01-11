import argparse
import random
from pathlib import Path

import numpy as np
import torch

from utils import ensure_dir, read_jsonl, set_seed


def main(args):
    set_seed(args.seed)
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

    for seq in sequences:
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

    torch.save(train, str(Path(args.out_dir) / "packed_train.pt"))
    torch.save(val, str(Path(args.out_dir) / "packed_val.pt"))
    print(f"train_planner={len(train['planner'])} train_renderer={len(train['renderer'])}")
    print(f"val_planner={len(val['planner'])} val_renderer={len(val['renderer'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
