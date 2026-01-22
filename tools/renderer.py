import math
import warnings

import torch
from torch import nn

from tools.data import BYTE_BOS, BYTE_EOS, BYTE_PAD, BYTE_VOCAB_SIZE


class Renderer(nn.Module):
    def __init__(
        self,
        K: int,
        V_list,
        d_resid: int = 128,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_len: int = 256,
        dropout: float = 0.1,
        vocab_size: int = BYTE_VOCAB_SIZE,
        d_ctx: int = 0,
    ):
        super().__init__()
        self.K = K
        self.V_list = list(V_list)
        self.d_resid = d_resid
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.d_ctx = d_ctx

        self.code_embeds = nn.ModuleList(
            [nn.Embedding(v, d_model) for v in self.V_list]
        )
        self.resid_proj = nn.Linear(d_resid, d_model)
        self.ctx_proj = nn.Linear(d_ctx, d_model) if d_ctx > 0 else None
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) / math.sqrt(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.len_head = nn.Linear(d_model, max_len + 1)

    def forward(self, codes: torch.Tensor, resid: torch.Tensor, ctx: torch.Tensor = None):
        cond = self.resid_proj(resid)
        for k in range(self.K):
            cond = cond + self.code_embeds[k](codes[:, k])
        if ctx is not None and self.ctx_proj is not None:
            cond = cond + self.ctx_proj(ctx)

        x = self.pos_emb.unsqueeze(0) + cond.unsqueeze(1)
        h = self.transformer(x)
        logits = self.out(h)
        pooled = h.mean(dim=1)
        len_logits = self.len_head(pooled)
        return logits, len_logits

    def generate(
        self,
        codes: torch.Tensor,
        resid: torch.Tensor,
        ctx: torch.Tensor = None,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        ban_repeats: bool = True,
        ascii_only: bool = True,
        min_len_bytes: int = 32,
        force_min_len: bool = False,
        return_logp: bool = False,
    ):
        logits, len_logits = self.forward(codes, resid, ctx=ctx)
        if ascii_only:
            allowed = torch.zeros(256, dtype=torch.bool, device=logits.device)
            allowed[0x20:0x7F] = True
            allowed[0x0A] = True
            allowed_mask = allowed
        else:
            allowed_mask = None
        length_logits = len_logits[:, 1:]
        min_len = max(0, min_len_bytes)
        if min_len > self.max_len:
            min_len = self.max_len
        enforce_min_len = min_len > 0
        if min_len > 1:
            length_mask = torch.arange(1, self.max_len + 1, device=length_logits.device)
            length_logits = length_logits.masked_fill(
                length_mask.unsqueeze(0) < min_len, -1e9
            )
        if sample:
            length_probs = torch.softmax(length_logits, dim=-1)
            lengths = torch.multinomial(length_probs, num_samples=1).squeeze(1) + 1
            if enforce_min_len or force_min_len:
                lengths = lengths.clamp_min(min_len)
            len_logp = torch.log(
                torch.gather(length_probs, 1, (lengths - 1).unsqueeze(1)).squeeze(1) + 1e-8
            )
            token_logits = logits if temperature == 1.0 else logits / temperature
            tokens = torch.zeros(
                token_logits.size(0),
                token_logits.size(1),
                dtype=torch.long,
                device=token_logits.device,
            )
            token_logp_steps = []
            prev = None
            penalty_tokens = None
            special_tokens = [BYTE_EOS]
            if enforce_min_len or force_min_len:
                special_tokens.extend([BYTE_PAD, BYTE_BOS])
            for i in range(token_logits.size(1)):
                step_logits = token_logits[:, i, :]
                if allowed_mask is not None:
                    step_logits = step_logits.clone()
                    step_logits[:, :256] = step_logits[:, :256].masked_fill(
                        ~allowed_mask, -1e9
                    )
                if BYTE_EOS < self.vocab_size and i < min_len:
                    step_logits = step_logits.clone()
                    step_logits[:, BYTE_EOS] = -1e9
                if enforce_min_len or force_min_len:
                    active = i < lengths
                    if active.any():
                        step_logits = step_logits.clone()
                        if self.vocab_size > 256 and i < min_len:
                            step_logits[active, 256:] = -1e9
                        for token in special_tokens:
                            if token < self.vocab_size:
                                step_logits[active, token] = -1e9
                if ban_repeats and penalty_tokens is not None:
                    step_logits = step_logits.clone()
                    mask = penalty_tokens >= 0
                    if mask.any():
                        step_logits[mask, penalty_tokens[mask]] -= 1.0
                if top_k and top_k > 0:
                    k = min(top_k, step_logits.size(-1))
                    topk_vals, _ = torch.topk(step_logits, k=k, dim=-1)
                    kth = topk_vals[:, -1].unsqueeze(-1)
                    step_logits = torch.where(
                        step_logits < kth, torch.full_like(step_logits, -1e9), step_logits
                    )
                step_probs = torch.softmax(step_logits, dim=-1)
                step_tokens = torch.multinomial(step_probs, num_samples=1).squeeze(1)
                tokens[:, i] = step_tokens
                if return_logp:
                    token_logp_steps.append(
                        torch.log(
                        torch.gather(step_probs, 1, step_tokens.unsqueeze(1)).squeeze(1)
                        + 1e-8
                    )
                    )
                if ban_repeats:
                    if prev is None:
                        penalty_tokens = None
                    else:
                        penalty_tokens = torch.where(
                            step_tokens == prev, step_tokens, torch.full_like(step_tokens, -1)
                        )
                    prev = step_tokens
            if (enforce_min_len or force_min_len) and BYTE_PAD < self.vocab_size:
                pad_mask = torch.arange(tokens.size(1), device=tokens.device)
                pad_mask = pad_mask.unsqueeze(0) >= lengths.unsqueeze(1)
                tokens = tokens.masked_fill(pad_mask, BYTE_PAD)
            if return_logp:
                token_logp = torch.stack(token_logp_steps, dim=1)
        else:
            lengths = length_logits.argmax(dim=-1) + 1
            if enforce_min_len or force_min_len:
                lengths = lengths.clamp_min(min_len)
            token_logits = logits
            if allowed_mask is not None:
                token_logits = token_logits.clone()
                token_logits[:, :, :256] = token_logits[:, :, :256].masked_fill(
                    ~allowed_mask, -1e9
                )
            if BYTE_EOS < self.vocab_size and min_len > 1:
                token_logits = token_logits.clone()
                token_logits[:, :min_len, BYTE_EOS] = -1e9
            if enforce_min_len or force_min_len:
                pos_mask = torch.arange(token_logits.size(1), device=token_logits.device)
                pos_mask = pos_mask.unsqueeze(0) < lengths.unsqueeze(1)
                idx_b, idx_t = pos_mask.nonzero(as_tuple=True)
                for token in (BYTE_EOS, BYTE_PAD, BYTE_BOS):
                    if token < self.vocab_size and idx_b.numel() > 0:
                        token_logits[idx_b, idx_t, token] = -1e9
                if self.vocab_size > 256 and min_len > 0 and idx_b.numel() > 0:
                    token_logits[idx_b, idx_t, 256:] = -1e9
            tokens = token_logits.argmax(dim=-1)
            if (enforce_min_len or force_min_len) and BYTE_PAD < self.vocab_size:
                pad_mask = torch.arange(tokens.size(1), device=tokens.device)
                pad_mask = pad_mask.unsqueeze(0) >= lengths.unsqueeze(1)
                tokens = tokens.masked_fill(pad_mask, BYTE_PAD)
            token_logp = None
            len_logp = None
        byte_counts = (tokens < 256).sum(dim=1)
        if (byte_counts == 0).any():
            fallback_len = max(min_len, 1)
            fallback_len = min(fallback_len, tokens.size(1))
            warnings.warn("renderer.generate: empty bytes; applying fallback sampling")
            bad = (byte_counts == 0).nonzero(as_tuple=True)[0]
            if return_logp and sample:
                if token_logp is None:
                    token_logp = torch.zeros(
                        tokens.size(0),
                        tokens.size(1),
                        dtype=logits.dtype,
                        device=logits.device,
                    )
            for b in bad.tolist():
                prev = None
                for i in range(fallback_len):
                    step_logits = logits[b, i, :256]
                    if allowed_mask is not None:
                        step_logits = step_logits.masked_fill(~allowed_mask, -1e9)
                    if temperature != 1.0:
                        step_logits = step_logits / temperature
                    if ban_repeats and prev is not None:
                        step_logits = step_logits.clone()
                        step_logits[prev] -= 1.0
                    if top_k and top_k > 0:
                        k = min(top_k, step_logits.size(-1))
                        topk_vals, _ = torch.topk(step_logits, k=k, dim=-1)
                        kth = topk_vals[-1]
                        step_logits = torch.where(
                            step_logits < kth,
                            torch.full_like(step_logits, -1e9),
                            step_logits,
                        )
                    step_probs = torch.softmax(step_logits, dim=-1)
                    token = torch.multinomial(step_probs, num_samples=1).squeeze(0)
                    tokens[b, i] = token
                    if return_logp and sample:
                        token_logp[b, i] = torch.log(step_probs[token] + 1e-8)
                    prev = int(token.item())
                if BYTE_PAD < self.vocab_size:
                    tokens[b, fallback_len:] = BYTE_PAD
                lengths[b] = max(int(lengths[b].item()), fallback_len)
            byte_counts = (tokens < 256).sum(dim=1)
            still_empty = (byte_counts == 0).nonzero(as_tuple=True)[0]
            if still_empty.numel() > 0:
                warnings.warn("renderer.generate: empty after fallback; using spaces")
                fill_len = max(min_len, 1)
                fill_len = min(fill_len, tokens.size(1))
                for b in still_empty.tolist():
                    tokens[b, :fill_len] = 32
                    if BYTE_PAD < self.vocab_size:
                        tokens[b, fill_len:] = BYTE_PAD
                    lengths[b] = max(int(lengths[b].item()), fill_len)
        if return_logp:
            return tokens, lengths, logits, len_logits, token_logp, len_logp
        return tokens, lengths, logits, len_logits
