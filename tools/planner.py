import math
import torch
from torch import nn


class Planner(nn.Module):
    def __init__(
        self,
        K: int,
        V_list,
        d_resid: int = 128,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_steps: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.V_list = list(V_list)
        self.d_resid = d_resid
        self.d_model = d_model
        self.max_steps = max_steps

        self.code_embeds = nn.ModuleList(
            [nn.Embedding(v, d_model) for v in self.V_list]
        )
        self.resid_proj = nn.Linear(d_resid, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_steps, d_model) / math.sqrt(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.code_heads = nn.ModuleList(
            [nn.Linear(d_model, v) for v in self.V_list]
        )
        self.resid_head = nn.Linear(d_model, d_resid)

    def forward(self, codes: torch.Tensor, resid: torch.Tensor, lengths: torch.Tensor = None):
        bsz, steps, _ = codes.shape
        if steps > self.max_steps:
            codes = codes[:, -self.max_steps :]
            resid = resid[:, -self.max_steps :]
            steps = self.max_steps

        x = self.resid_proj(resid)
        for k in range(self.K):
            x = x + self.code_embeds[k](codes[:, :, k])
        x = x + self.pos_emb[:steps].unsqueeze(0)

        if lengths is not None:
            pad = torch.arange(steps, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            pad = None

        causal = torch.triu(
            torch.full((steps, steps), float("-inf"), device=x.device), diagonal=1
        )
        h = self.transformer(x, mask=causal, src_key_padding_mask=pad)

        code_logits = [head(h) for head in self.code_heads]
        pred_resid = self.resid_head(h)
        return code_logits, pred_resid, h
