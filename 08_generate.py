import argparse
import os
from pathlib import Path

import torch

from tools.data import BYTE_EOS, BYTE_PAD, BYTE_VOCAB_SIZE, bytes_to_text
from tools.encoder import ByteEncoder
from tools.planner import Planner
from tools.renderer import Renderer
from tools.rvq import load_rvq
from utils import setup_runtime, split_sentences


def main(args):
    logger, device = setup_runtime(
        "08_generate",
        device=args.device,
        threads=args.threads,
    )

    enc_ckpt = torch.load(args.encoder, map_location=device)
    encoder = ByteEncoder(**enc_ckpt["config"])
    encoder.load_state_dict(enc_ckpt["model"])
    encoder.to(device)
    encoder.eval()

    rvq = load_rvq(args.rvq, device=device)

    plan_ckpt = torch.load(args.planner, map_location=device)
    planner = Planner(**plan_ckpt["config"])
    planner.load_state_dict(plan_ckpt["model"])
    planner.to(device)
    planner.eval()

    rend_path = args.renderer
    if not os.path.exists(rend_path):
        fallback = "out/renderer_latest.pt"
        if rend_path == "out/renderer.pt" and os.path.exists(fallback):
            rend_path = fallback
        else:
            raise FileNotFoundError(rend_path)
    rend_ckpt = torch.load(rend_path, map_location=device)
    renderer = Renderer(**rend_ckpt["config"], vocab_size=BYTE_VOCAB_SIZE)
    renderer.load_state_dict(rend_ckpt["model"])
    renderer.to(device)
    renderer.eval()

    sentences = split_sentences(args.prompt)
    if not sentences:
        raise ValueError("empty prompt")
    prompt_sent_count = len(sentences)

    with torch.no_grad():
        codes_seq = []
        resid_seq = []
        last_raw = None
        for sent in sentences:
            ids = encoder_input_ids(sent, enc_ckpt["config"]["max_len"], device)
            emb = encoder(ids)
            codes, _, resid = rvq.encode(emb)
            codes_seq.append(codes.squeeze(0))
            resid_seq.append(resid.squeeze(0))

        for _ in range(args.steps):
            codes_t = torch.stack(codes_seq).unsqueeze(0).to(device)
            resid_t = torch.stack(resid_seq).unsqueeze(0).to(device)
            lengths = torch.tensor([codes_t.size(1)], device=device)
            code_logits, pred_resid, _ = planner(codes_t, resid_t, lengths=lengths)
            next_codes = torch.stack([logits[0, -1].argmax(dim=-1) for logits in code_logits])
            next_resid = pred_resid[0, -1]

            text, last_raw = render_sentence(renderer, next_codes, next_resid, args, logger)
            sentences.append(text)
            codes_seq.append(next_codes)
            resid_seq.append(next_resid)

    gen_text = " ".join(sentences[prompt_sent_count:])
    out_text = f"{args.prompt}\n{gen_text}"
    Path(args.out).write_text(out_text, encoding="utf-8")
    if args.debug_dump and last_raw is None:
        Path("out/gen.bin").write_bytes(b"")
    logger.info("saved=%s", args.out)
    print(out_text)


def encoder_input_ids(text: str, max_len: int, device):
    data = text.encode("utf-8", errors="ignore")
    ids = list(data)[:max_len]
    if len(ids) < max_len:
        ids.append(BYTE_EOS)
    if len(ids) < max_len:
        ids += [BYTE_PAD] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def render_sentence(renderer, codes, resid, args, logger):
    def _generate(temp, top_k, sample_flag):
        tokens, lengths, _, _ = renderer.generate(
            codes.unsqueeze(0),
            resid.unsqueeze(0),
            ctx=None,
            sample=sample_flag,
            temperature=temp,
            top_k=top_k,
            ban_repeats=args.ban_repeats,
            ascii_only=args.ascii_only,
            min_len_bytes=args.min_len_bytes,
        )
        row = tokens[0, : lengths.item()].tolist()
        raw = bytes([t for t in row if 0 <= t < 256])
        text = bytes_to_text(raw)
        return raw, text

    with torch.no_grad():
        raw_bytes, text = _generate(args.temperature, args.top_k, args.sample)
        if raw_bytes is None:
            raw_bytes = b""
            text = ""
        if len(raw_bytes) == 0 and args.min_len_bytes > 0:
            print(
                "warning: empty generation; retrying with temperature=1.2 and top_k>=40"
            )
            raw_bytes, text = _generate(1.2, max(args.top_k, 40), True)
        if len(raw_bytes) == 0 and args.min_len_bytes > 0:
            print("warning: empty generation after retry")
        logger.info("gen_raw_len=%d", len(raw_bytes))
        if args.debug_dump:
            raw_len = len(raw_bytes)
            hex_bytes = " ".join(f"{b:02x}" for b in raw_bytes[:64])
            preview = "".join(ch if ch.isprintable() else "." for ch in text)
            print(f"raw_len_bytes={raw_len}")
            print(f"raw_bytes_hex={hex_bytes}")
            print(f"decoded_text={text!r} len={len(text)}")
            print(f"preview={preview}")
            if args.ascii_only:
                allowed_count = (0x7E - 0x20) + 1 + 1
            else:
                allowed_count = 256
            print(f"allowed_bytes={allowed_count}")
            Path("out/gen.bin").write_bytes(raw_bytes)
    return text, raw_bytes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--encoder", default="out/enc.pt")
    ap.add_argument("--rvq", default="out/rvq.pt")
    ap.add_argument("--planner", default="out/planner.pt")
    ap.add_argument("--renderer", default="out/renderer.pt")
    ap.add_argument("--out", default="out/gen.txt")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument(
        "--no-ban-repeats",
        dest="ban_repeats",
        action="store_false",
        help="disable simple anti-run penalty while sampling",
    )
    ap.set_defaults(ban_repeats=True)
    ap.add_argument(
        "--ascii_only",
        "--ascii-only",
        dest="ascii_only",
        action="store_true",
        help="restrict sampling to printable ASCII plus newline",
    )
    ap.add_argument(
        "--no-ascii-only",
        dest="ascii_only",
        action="store_false",
        help="allow non-ascii bytes in sampling",
    )
    ap.set_defaults(ascii_only=True)
    ap.add_argument("--min_len_bytes", type=int, default=32)
    ap.add_argument("--debug_dump", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    main(args)
