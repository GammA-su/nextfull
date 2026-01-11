import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tools.rvq import load_rvq
from utils import ensure_dir


def main(args):
    ensure_dir(args.out_dir)
    device = torch.device(args.device)

    emb = np.load(args.emb).astype(np.float32)
    rvq = load_rvq(args.rvq, device=device)

    codes_out = []
    resid_out = []
    bs = args.batch_size
    for i in tqdm(range(0, emb.shape[0], bs), desc="encode"):
        batch = torch.tensor(emb[i : i + bs], device=device)
        with torch.no_grad():
            codes, q, resid = rvq.encode(batch)
        codes_out.append(codes.cpu().numpy().astype(np.int32))
        resid_out.append(resid.cpu().numpy().astype(np.float32))

    codes_mat = np.concatenate(codes_out, axis=0)
    resid_mat = np.concatenate(resid_out, axis=0)

    np.save(str(Path(args.out_dir) / "sent_codes.npy"), codes_mat)
    np.save(str(Path(args.out_dir) / "sent_resid.npy"), resid_mat)
    print(f"saved={Path(args.out_dir) / 'sent_codes.npy'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/sent_emb.npy")
    ap.add_argument("--rvq", default="out/rvq.pt")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    main(args)
