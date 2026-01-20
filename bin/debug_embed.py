import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.data import BYTE_PAD, text_to_bytes
from tools.encoder import ByteEncoder


def encode_texts(texts, encoder: ByteEncoder, device: str):
    max_len = encoder.max_len
    batch = []
    for text in texts:
        ids = text_to_bytes(text, max_len=max_len, add_eos=True)
        if len(ids) < max_len:
            ids = ids + [BYTE_PAD] * (max_len - len(ids))
        batch.append(ids)
    ids = torch.tensor(batch, dtype=torch.long, device=device)
    with torch.inference_mode():
        emb = encoder(ids)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", required=True, help="text to embed (repeatable)")
    ap.add_argument("--encoder", default="out/enc.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.encoder, map_location=device)
    encoder = ByteEncoder(**ckpt["config"])
    encoder.load_state_dict(ckpt["model"])
    encoder.to(device)
    encoder.eval()

    texts = args.text
    emb = encode_texts(texts, encoder, device)
    norms = emb.norm(dim=-1).cpu()
    emb_cpu = emb.cpu()

    for i, text in enumerate(texts):
        vec = emb_cpu[i]
        first_vals = " ".join(f"{v:.4f}" for v in vec[:8])
        print(f"text[{i}] norm={norms[i]:.4f} first8={first_vals}")

    if len(texts) >= 2:
        emb_norm = emb_cpu / (emb_cpu.norm(dim=-1, keepdim=True) + 1e-9)
        cos = emb_norm @ emb_norm.t()
        print("cosine_matrix:")
        for i in range(cos.size(0)):
            row = " ".join(f"{v:.4f}" for v in cos[i].tolist())
            print(row)


if __name__ == "__main__":
    main()
