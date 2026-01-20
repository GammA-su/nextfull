import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.data import BYTE_PAD, text_to_bytes
from tools.encoder import ByteEncoder
from tools.reward import reward


def read_inputs(gen_arg: str, target_arg: str):
    if gen_arg != "-" and target_arg != "-":
        return gen_arg, target_arg

    data = sys.stdin.read()
    data = data.rstrip("\n")
    if gen_arg == "-" and target_arg == "-":
        lines = data.splitlines()
        if len(lines) < 2:
            raise ValueError("stdin must contain at least two lines for --gen and --target")
        return lines[0], lines[1]
    if gen_arg == "-":
        return data, target_arg
    return gen_arg, data


def encode_text(text: str, max_len: int, device):
    ids = text_to_bytes(text, max_len=max_len, add_eos=True)
    if len(ids) < max_len:
        ids = ids + [BYTE_PAD] * (max_len - len(ids))
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    return tensor, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generated text or '-' for stdin")
    ap.add_argument("--target", required=True, help="target text or '-' for stdin")
    ap.add_argument("--encoder", default="out/enc.pt")
    ap.add_argument("--min_len_bytes", type=int, default=32)
    args = ap.parse_args()

    gen_text, target_text = read_inputs(args.gen, args.target)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.encoder, map_location=device)
    encoder = ByteEncoder(**ckpt["config"])
    encoder.load_state_dict(ckpt["model"])
    encoder.to(device)
    encoder.eval()

    max_len = ckpt["config"]["max_len"]
    gen_ids, gen_len = encode_text(gen_text, max_len, device)
    tgt_ids, _ = encode_text(target_text, max_len, device)

    with torch.inference_mode():
        gen_emb = encoder(gen_ids)
        tgt_emb = encoder(tgt_ids)

    norm_gen = float(gen_emb.norm(dim=-1)[0])
    norm_tgt = float(tgt_emb.norm(dim=-1)[0])
    max_abs_diff = float((gen_emb - tgt_emb).abs().max())
    print(f"norm_gen={norm_gen:.4f}")
    print(f"norm_tgt={norm_tgt:.4f}")
    print(f"max_abs_diff={max_abs_diff:.6f}")

    gen_tokens = text_to_bytes(gen_text, max_len=max_len, add_eos=True)
    lengths = torch.tensor([gen_len], dtype=torch.long, device=device)

    alpha_len = 0.05
    beta_rep = 0.1
    invalid_penalty = 0.5

    final_reward, comps = reward(
        gen_emb,
        tgt_emb,
        [gen_tokens],
        lengths,
        max_len,
        alpha_len,
        beta_rep,
        invalid_penalty,
        min_len_bytes=args.min_len_bytes,
    )

    cosine = float(comps["cosine"][0])
    len_pen = float(comps["len_pen"][0])
    rep_pen = float(comps["rep_pen"][0])
    inv_pen = float(comps["inv_pen"][0])
    spam = int(comps["spam"][0].item() > 0)
    reward_val = float(final_reward[0])

    print(f"cosine={cosine:.4f}")
    print(f"len_pen={len_pen:.4f}")
    print(f"rep_pen={rep_pen:.4f}")
    print(f"inv_pen={inv_pen:.4f}")
    print(f"spam={spam}")
    print(f"reward={reward_val:.4f}")


if __name__ == "__main__":
    main()
