import argparse
import itertools
import time
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

    logger.info("load: rvq=%s", args.rvq)
    t0 = time.time()
    rvq = load_rvq(args.rvq, device=device)
    logger.info("load: rvq done (%.2fs)", time.time() - t0)

    logger.info("load: train_data=%s", args.train_data)
    t0 = time.time()
    pack = torch.load(args.train_data, map_location="cpu")
    logger.info(
        "load: train_data done (%.2fs) renderer_samples=%d",
        time.time() - t0,
        len(pack.get("renderer", [])),
    )
    train_ds = RendererDataset(pack["renderer"])
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, max(1, args.threads // 2))
    pin_memory = device.type == "cuda"
    logger.info(
        "data: loader batch_size=%d num_workers=%d pin_memory=%s",
        args.batch_size,
        num_workers,
        pin_memory,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_renderer(b, max_len=args.max_len),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    logger.info("load: encoder=%s", args.encoder)
    t0 = time.time()
    enc_ckpt = torch.load(args.encoder, map_location=device)
    encoder = ByteEncoder(**enc_ckpt["config"])
    encoder.load_state_dict(enc_ckpt["model"])
    encoder.to(device)
    encoder.eval()
    logger.info("load: encoder done (%.2fs)", time.time() - t0)

    d_resid = pack["renderer"][0]["resid"].shape[0]
    logger.info("init: renderer d_resid=%d", d_resid)
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
    logger.info("init: renderer done")

    def _atomic_save(obj, path: Path):
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(obj, str(tmp_path))
        tmp_path.replace(path)

    def _move_optimizer_state(opt, device):
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _save_checkpoint(path: Path, model, opt, epoch, step, global_step, config):
        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "config": config,
            "rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        _atomic_save(ckpt, path)

    start_epoch = 0
    start_step = 0
    global_step = 0
    resume_path = None
    if args.resume and args.resume.lower() != "none":
        if args.resume == "auto":
            candidate = Path(args.out_dir) / "renderer_latest.pt"
            if candidate.exists():
                resume_path = candidate
        else:
            resume_path = Path(args.resume)
    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            _move_optimizer_state(opt, device)
        start_epoch = int(ckpt.get("epoch", 0))
        start_step = int(ckpt.get("step", 0))
        global_step = int(ckpt.get("global_step", 0))
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            try:
                if torch.is_tensor(rng_state):
                    rng_state = rng_state.detach().to("cpu")
                torch.set_rng_state(rng_state)
            except Exception:
                logger.warning("resume: rng_state invalid; skipping restore")
        if device.type == "cuda":
            cuda_state = ckpt.get("cuda_rng_state")
            if cuda_state is not None:
                try:
                    if isinstance(cuda_state, (list, tuple)):
                        cuda_state = [
                            t.detach().to("cpu") if torch.is_tensor(t) else t
                            for t in cuda_state
                        ]
                    torch.cuda.set_rng_state_all(cuda_state)
                except Exception:
                    logger.warning("resume: cuda_rng_state invalid; skipping restore")
        logger.info(
            "resume: path=%s epoch=%d step=%d global_step=%d",
            resume_path,
            start_epoch,
            start_step,
            global_step,
        )

    config = {
        "K": rvq.K,
        "V_list": rvq.V_list,
        "d_resid": d_resid,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_len": args.max_len,
        "dropout": args.dropout,
    }

    loader_len = len(loader)
    resume_step = start_step
    for epoch in range(start_epoch, args.epochs):
        skip_steps = resume_step if epoch == start_epoch else 0
        logger.info(
            "epoch=%d train_batches=%d batch_size=%d",
            epoch,
            loader_len,
            args.batch_size,
        )
        model.train()
        train_iter = iter(loader)
        skipped = 0
        if skip_steps > 0:
            t_skip = time.time()
            for _ in range(skip_steps):
                try:
                    next(train_iter)
                except StopIteration:
                    break
                skipped += 1
            logger.info(
                "resumed epoch=%d skipped_batches=%d (%.2fs)",
                epoch,
                skipped,
                time.time() - t_skip,
            )
        remaining = max(loader_len - skipped, 0)
        if remaining > 0:
            logger.info("data: waiting_for_first_batch epoch=%d", epoch)
            t_first = time.time()
            try:
                first_batch = next(train_iter)
            except StopIteration:
                first_batch = None
                remaining = 0
            else:
                dt = time.time() - t_first
                codes0, resid0, emb0, bytes0, lengths0 = first_batch
                logger.info(
                    "data: first_batch_ready epoch=%d time=%.2fs codes=%s resid=%s emb=%s bytes=%s lengths=%s",
                    epoch,
                    dt,
                    tuple(codes0.shape),
                    tuple(resid0.shape),
                    tuple(emb0.shape),
                    tuple(bytes0.shape),
                    tuple(lengths0.shape),
                )
                train_iter = itertools.chain([first_batch], train_iter)
        else:
            logger.warning("data: no_batches epoch=%d", epoch)
        pbar = tqdm(train_iter, desc=f"epoch {epoch}", total=remaining)
        start = time.time()
        last_log = start
        last_time_log = start
        running_loss = 0.0
        running_reward = 0.0
        running_adv = 0.0
        running_len = 0.0
        running_len_pen = 0.0
        running_rep_pen = 0.0
        running_inv_pen = 0.0
        time_loss = 0.0
        time_reward = 0.0
        time_adv = 0.0
        time_len = 0.0
        time_len_pen = 0.0
        time_rep_pen = 0.0
        time_inv_pen = 0.0
        time_steps = 0
        step_in_epoch = 0
        last_step = skip_steps
        for step_in_epoch, (codes, resid, tgt_emb, _, _) in enumerate(pbar, start=1):
            step = skip_steps + step_in_epoch
            last_step = step
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
            mean_reward = float(samp_reward.mean())
            mean_adv = float(adv.mean())
            mean_len = float(samp_len.float().mean())
            mean_len_pen = float(samp_len_pen.mean())
            mean_rep_pen = float(samp_rep_pen.mean())
            mean_inv_pen = float(samp_inv_pen.mean())
            pbar.set_postfix(loss=float(loss), reward=mean_reward)
            running_loss += float(loss)
            running_reward += mean_reward
            running_adv += mean_adv
            running_len += mean_len
            running_len_pen += mean_len_pen
            running_rep_pen += mean_rep_pen
            running_inv_pen += mean_inv_pen
            time_loss += float(loss)
            time_reward += mean_reward
            time_adv += mean_adv
            time_len += mean_len
            time_len_pen += mean_len_pen
            time_rep_pen += mean_rep_pen
            time_inv_pen += mean_inv_pen
            time_steps += 1
            global_step += 1
            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_path = Path(args.out_dir) / "renderer_latest.pt"
                _save_checkpoint(
                    ckpt_path,
                    model,
                    opt,
                    epoch,
                    step,
                    global_step,
                    config,
                )
                logger.info("checkpoint=%s", ckpt_path)
            if args.log_every > 0 and step % args.log_every == 0:
                now = time.time()
                step_time = now - last_log
                rate = (args.log_every / step_time) if step_time > 0 else 0.0
                logger.info(
                    "epoch=%d step=%d/%d rate=%.2f steps/s loss=%.4f reward=%.4f adv=%.4f len=%.1f len_pen=%.4f rep_pen=%.4f inv_pen=%.4f",
                    epoch,
                    step,
                    len(loader),
                    rate,
                    running_loss / args.log_every,
                    running_reward / args.log_every,
                    running_adv / args.log_every,
                    running_len / args.log_every,
                    running_len_pen / args.log_every,
                    running_rep_pen / args.log_every,
                    running_inv_pen / args.log_every,
                )
                running_loss = 0.0
                running_reward = 0.0
                running_adv = 0.0
                running_len = 0.0
                running_len_pen = 0.0
                running_rep_pen = 0.0
                running_inv_pen = 0.0
                last_log = now
            if args.log_time_every > 0:
                now = time.time()
                if now - last_time_log >= args.log_time_every:
                    rate = (time_steps / (now - last_time_log)) if time_steps > 0 else 0.0
                    denom = max(1, time_steps)
                    logger.info(
                        "epoch=%d step=%d/%d rate=%.2f steps/s loss=%.4f reward=%.4f adv=%.4f len=%.1f len_pen=%.4f rep_pen=%.4f inv_pen=%.4f",
                        epoch,
                        step,
                        len(loader),
                        rate,
                        time_loss / denom,
                        time_reward / denom,
                        time_adv / denom,
                        time_len / denom,
                        time_len_pen / denom,
                        time_rep_pen / denom,
                        time_inv_pen / denom,
                    )
                    time_loss = 0.0
                    time_reward = 0.0
                    time_adv = 0.0
                    time_len = 0.0
                    time_len_pen = 0.0
                    time_rep_pen = 0.0
                    time_inv_pen = 0.0
                    time_steps = 0
                    last_time_log = now
        if last_step > skip_steps:
            ckpt_path = Path(args.out_dir) / "renderer_latest.pt"
            _save_checkpoint(
                ckpt_path,
                model,
                opt,
                epoch,
                last_step,
                global_step,
                config,
            )
            logger.info("checkpoint=%s", ckpt_path)
        resume_step = 0
    final_ckpt = {
        "model": model.state_dict(),
        "config": config,
    }
    out_path = Path(args.out_dir) / "renderer.pt"
    _atomic_save(final_ckpt, out_path)
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
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--log_time_every", type=int, default=30)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument(
        "--resume",
        default="auto",
        help="checkpoint path, 'auto' for renderer_latest.pt in out_dir, or 'none'",
    )
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
