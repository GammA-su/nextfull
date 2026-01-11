import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tools.rvq import load_rvq
from utils import detect_faiss, ensure_dir, setup_runtime


def build_faiss_indexes(codebooks, use_gpu: bool):
    import faiss

    res = faiss.StandardGpuResources() if use_gpu else None
    indexes = []
    cb_np_list = []
    for cb in codebooks:
        cb_np = cb.detach().cpu().numpy().astype(np.float32, copy=False)
        index = faiss.IndexFlatL2(cb_np.shape[1])
        if use_gpu:
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(cb_np)
        indexes.append(index)
        cb_np_list.append(cb_np)
    return indexes, cb_np_list


def faiss_encode_batch(batch_np, indexes, cb_np_list):
    residual = batch_np.astype(np.float32, copy=False)
    codes = []
    for index, cb_np in zip(indexes, cb_np_list):
        _, idx = index.search(residual, 1)
        code = idx[:, 0]
        q = cb_np[code]
        codes.append(code)
        residual = residual - q
    codes = np.stack(codes, axis=1)
    return codes, residual


def main(args):
    logger, device = setup_runtime(
        "04b_encode_codes",
        device=args.device,
        threads=args.threads,
    )
    ensure_dir(args.out_dir)

    emb = np.load(args.emb).astype(np.float32)
    rvq = load_rvq(args.rvq, device=device)

    faiss_available, faiss_gpu_count = detect_faiss()
    use_faiss = (not args.no_faiss) and faiss_available and faiss_gpu_count > 0
    if use_faiss and rvq.usage_balance_w > 0:
        logger.warning(
            "faiss encode ignores usage_balance_w=%.3f", rvq.usage_balance_w
        )
    if use_faiss:
        logger.info("faiss_encode=true faiss_gpu=true gpus=%d", faiss_gpu_count)
        indexes, cb_np_list = build_faiss_indexes(rvq.codebooks, use_gpu=True)
    else:
        logger.info("faiss_encode=false (fallback to torch)")

    codes_out = []
    resid_out = []
    bs = args.batch_size
    for i in tqdm(range(0, emb.shape[0], bs), desc="encode"):
        batch_np = emb[i : i + bs]
        if use_faiss:
            codes, resid = faiss_encode_batch(batch_np, indexes, cb_np_list)
            codes_out.append(codes.astype(np.int32))
            resid_out.append(resid.astype(np.float32))
        else:
            batch = torch.tensor(batch_np, device=device)
            with torch.no_grad():
                codes, _, resid = rvq.encode(batch)
            codes_out.append(codes.cpu().numpy().astype(np.int32))
            resid_out.append(resid.cpu().numpy().astype(np.float32))

    codes_mat = np.concatenate(codes_out, axis=0)
    resid_mat = np.concatenate(resid_out, axis=0)

    np.save(str(Path(args.out_dir) / "sent_codes.npy"), codes_mat)
    np.save(str(Path(args.out_dir) / "sent_resid.npy"), resid_mat)
    logger.info("saved=%s", Path(args.out_dir) / "sent_codes.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/sent_emb.npy")
    ap.add_argument("--rvq", default="out/rvq.pt")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no_faiss", action="store_true")
    args = ap.parse_args()
    main(args)
