import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.data import (
    BYTE_EOS,
    BYTE_PAD,
    BYTE_VOCAB_SIZE,
    RendererDataset,
    collate_renderer,
    tokens_to_bytes,
)
from tools.encoder import ByteEncoder
from tools.renderer import Renderer
from tools.reward import (
    batch_repetition_penalty,
    cosine_reward,
    length_penalty,
    utf8_invalid_penalty,
)
from tools.rvq import load_rvq
from utils import ensure_dir, setup_runtime


def sample_lengths(len_logits):
    logits = len_logits[:, 1:]
    probs = F.softmax(logits, dim=-1)
    lengths = torch.multinomial(probs, num_samples=1).squeeze(1) + 1
    logp = torch.log(torch.gather(probs, 1, (lengths - 1).unsqueeze(1)).squeeze(1) + 1e-8)
    return lengths, logp


def greedy_lengths(len_logits):
    logits = len_logits[:, 1:]
    lengths = logits.argmax(dim=-1) + 1
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(torch.gather(probs, 1, (lengths - 1).unsqueeze(1)).squeeze(1) + 1e-8)
    return lengths, logp


def sample_tokens(logits, temperature: float):
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1))
    logp = torch.log(
        torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1) + 1e-8
    )
    return tokens, logp


def greedy_tokens(logits):
    tokens = logits.argmax(dim=-1)
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(
        torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1) + 1e-8
    )
    return tokens, logp


def build_encoder_inputs(batch_tokens, max_len):
    out = []
    for tokens in batch_tokens:
        ids = tokens[:max_len]
        if len(ids) < max_len:
            ids = ids + [BYTE_EOS]
        if len(ids) < max_len:
            ids = ids + [BYTE_PAD] * (max_len - len(ids))
        out.append(ids[:max_len])
    return torch.tensor(out, dtype=torch.long)


def decode_batch(tokens, lengths):
    batch = []
    for row, L in zip(tokens.tolist(), lengths.tolist()):
        raw = row[:L]
        batch.append(tokens_to_bytes(raw))
    return batch


def main(args):
    logger, device = setup_runtime(
        "07_train_renderer_reward",
        device=args.device,
        threads=args.threads,
        seed=args.seed,
    )
    ensure_dir(args.out_dir)

    rvq = load_rvq(args.rvq, device=device)
    pack = torch.load(args.train_data, map_location="cpu")
    train_ds = RendererDataset(pack["renderer"])
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, max(1, args.threads // 2))
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_renderer(b, max_len=args.max_len),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    enc_ckpt = torch.load(args.encoder, map_location=device)
    encoder = ByteEncoder(**enc_ckpt["config"])
    encoder.load_state_dict(enc_ckpt["model"])
    encoder.to(device)
    encoder.eval()

    d_resid = pack["renderer"][0]["resid"].shape[0]
    model = Renderer(
        K=rvq.K,
        V_list=rvq.V_list,
        d_resid=d_resid,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=args.dropout,
        vocab_size=BYTE_VOCAB_SIZE,
        d_ctx=0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for codes, resid, tgt_emb, _, _ in pbar:
            codes = codes.to(device, non_blocking=True)
            resid = resid.to(device, non_blocking=True)
            tgt_emb = tgt_emb.to(device, non_blocking=True)

            logits, len_logits = model(codes, resid, ctx=None)

            samp_len, samp_len_logp = sample_lengths(len_logits)
            samp_tokens, samp_token_logp = sample_tokens(logits, args.temperature)

            greedy_len, _ = greedy_lengths(len_logits)
            greedy_tokens_ids, _ = greedy_tokens(logits)

            samp_tokens_list = decode_batch(samp_tokens, samp_len)
            greedy_tokens_list = decode_batch(greedy_tokens_ids, greedy_len)

            with torch.no_grad():
                samp_ids = build_encoder_inputs(samp_tokens_list, max_len=args.max_bytes).to(
                    device, non_blocking=True
                )
                greedy_ids = build_encoder_inputs(greedy_tokens_list, max_len=args.max_bytes).to(
                    device, non_blocking=True
                )
                samp_emb = encoder(samp_ids)
                greedy_emb = encoder(greedy_ids)

            samp_reward = cosine_reward(samp_emb, tgt_emb)
            greedy_reward = cosine_reward(greedy_emb, tgt_emb)

            samp_len_pen = length_penalty(samp_len, args.alpha_len, args.max_len)
            samp_rep_pen = batch_repetition_penalty(samp_tokens_list).to(device) * args.beta_rep
            samp_inv_pen = utf8_invalid_penalty(samp_tokens_list, args.invalid_penalty).to(device)

            greedy_len_pen = length_penalty(greedy_len, args.alpha_len, args.max_len)
            greedy_rep_pen = batch_repetition_penalty(greedy_tokens_list).to(device) * args.beta_rep
            greedy_inv_pen = utf8_invalid_penalty(greedy_tokens_list, args.invalid_penalty).to(device)

            samp_reward = samp_reward - samp_len_pen - samp_rep_pen - samp_inv_pen
            greedy_reward = greedy_reward - greedy_len_pen - greedy_rep_pen - greedy_inv_pen

            adv = samp_reward - greedy_reward
            adv = adv.clamp(min=-args.adv_clip, max=args.adv_clip)

            token_logp = samp_token_logp
            mask = (
                torch.arange(token_logp.size(1), device=device).unsqueeze(0)
                < samp_len.unsqueeze(1)
            )
            token_logp = (token_logp * mask).sum(dim=1)
            logp = token_logp + samp_len_logp

            loss = -(adv.detach() * logp).mean()

            if args.entropy_bonus > 0:
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = loss - args.entropy_bonus * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            pbar.set_postfix(loss=float(loss), reward=float(samp_reward.mean()))

    ckpt = {
        "model": model.state_dict(),
        "config": {
            "K": rvq.K,
            "V_list": rvq.V_list,
            "d_resid": d_resid,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "max_len": args.max_len,
            "dropout": args.dropout,
        },
    }
    out_path = Path(args.out_dir) / "renderer.pt"
    torch.save(ckpt, str(out_path))
    logger.info("saved=%s", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", default="data/packed_train.pt")
    ap.add_argument("--encoder", default="out/enc.pt")
    ap.add_argument("--rvq", default="out/rvq.pt")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha_len", type=float, default=0.05)
    ap.add_argument("--beta_rep", type=float, default=0.1)
    ap.add_argument("--invalid_penalty", type=float, default=0.5)
    ap.add_argument("--adv_clip", type=float, default=2.0)
    ap.add_argument("--entropy_bonus", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()
    main(args)
