import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tools.data import BYTE_PAD, text_to_bytes
from tools.encoder import ByteEncoder
from utils import ensure_dir, read_jsonl, setup_runtime, write_jsonl


def main(args):
    logger, device = setup_runtime(
        "03_embed_sentences",
        device=args.device,
        threads=args.threads,
    )
    ensure_dir(args.out_dir)

    sentences = read_jsonl(str(Path(args.data_dir) / "sentences.jsonl"))
    sentences = sorted(sentences, key=lambda x: x["sid"])
    logger.info("sentences=%d", len(sentences))

    ckpt = torch.load(args.encoder, map_location=device)
    model = ByteEncoder(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    total = len(sentences)
    logger.info(
        "embed: batch_size=%d max_bytes=%d amp=%s log_every=%d",
        args.batch_size,
        args.max_bytes,
        args.amp,
        args.log_every,
    )

    all_emb = []
    meta = []
    batch = []
    processed = 0
    start = time.time()
    last_log = start
    next_log = args.log_every if args.log_every > 0 else None
    for item in sentences:
        ids = text_to_bytes(item["text"], max_len=args.max_bytes, add_eos=True)
        if len(ids) < args.max_bytes:
            ids = ids + [BYTE_PAD] * (args.max_bytes - len(ids))
        batch.append(ids)
        meta.append({"sid": item["sid"], "doc_id": item["doc_id"], "sent_idx": item["sent_idx"]})
        if len(batch) >= args.batch_size:
            emb = run_batch(model, batch, device, amp=args.amp)
            all_emb.append(emb)
            processed += len(batch)
            batch = []
            if next_log is not None and processed >= next_log:
                now = time.time()
                elapsed = now - start
                step = now - last_log
                rate = (args.log_every / step) if step > 0 else 0.0
                avg_rate = (processed / elapsed) if elapsed > 0 else 0.0
                remaining = total - processed
                eta = (remaining / avg_rate) if avg_rate > 0 else 0.0
                logger.info(
                    "progress: %d/%d (%.2f%%) rate=%.1f/s avg=%.1f/s eta=%.1fs",
                    processed,
                    total,
                    100.0 * processed / total if total else 0.0,
                    rate,
                    avg_rate,
                    eta,
                )
                last_log = now
                next_log += args.log_every

    if batch:
        emb = run_batch(model, batch, device, amp=args.amp)
        all_emb.append(emb)
        processed += len(batch)

    emb_mat = np.concatenate(all_emb, axis=0)
    np.save(str(Path(args.out_dir) / "sent_emb.npy"), emb_mat)
    write_jsonl(str(Path(args.out_dir) / "sent_meta.jsonl"), meta)
    logger.info("saved=%s", Path(args.out_dir) / "sent_emb.npy")


def run_batch(model, batch, device, amp: bool = False):
    with torch.inference_mode():
        ids = torch.tensor(batch, dtype=torch.long, device=device)
        if amp and getattr(device, "type", None) == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = model(ids).cpu().numpy()
        else:
            emb = model(ids).cpu().numpy()
    return emb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--encoder", default="out/enc.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_bytes", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=50000)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    main(args)
