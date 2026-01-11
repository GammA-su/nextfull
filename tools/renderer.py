import math
import torch
from torch import nn

from tools.data import BYTE_VOCAB_SIZE


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
