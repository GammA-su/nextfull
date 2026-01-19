import math
import torch
from torch import nn
from torch.nn import functional as F

from tools.data import BYTE_PAD, BYTE_VOCAB_SIZE


class ByteEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 256,
        d_emb: int = 128,
        dropout: float = 0.1,
        vocab_size: int = BYTE_VOCAB_SIZE,
        pad_id: int = BYTE_PAD,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.max_len = max_len
        self.pad_id = pad_id
        self.config = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "max_len": max_len,
            "d_emb": d_emb,
            "dropout": dropout,
        }
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) / math.sqrt(d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_emb)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of byte ids [B, L] into sentence embeddings [B, d_emb]."""
        if ids.size(1) > self.max_len:
            ids = ids[:, : self.max_len]
        mask = ids.eq(self.pad_id)
        x = self.token_emb(ids)
        pos = self.pos_emb[: ids.size(1)].unsqueeze(0)
        x = x + pos
        x = self.encoder(x, src_key_padding_mask=mask)
        keep = (~mask).float().unsqueeze(-1)
        denom = keep.sum(dim=1).clamp(min=1.0)
        pooled = (x * keep).sum(dim=1) / denom
        emb = self.proj(pooled)
        emb = F.normalize(emb, dim=-1)
        return emb

    def save(self, path: str) -> None:
        ckpt = {"model": self.state_dict(), "config": self.config}
        torch.save(ckpt, path)


def load_encoder(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model = ByteEncoder(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def encode_texts(model: ByteEncoder, texts, max_len: int, device: str):
    from tools.data import text_to_bytes

    batch = []
    for text in texts:
        ids = text_to_bytes(text, max_len=max_len, add_eos=True)
        if len(ids) < max_len:
            ids = ids + [BYTE_PAD] * (max_len - len(ids))
        batch.append(ids)
    ids = torch.tensor(batch, dtype=torch.long, device=device)
    with torch.inference_mode():
        return model(ids)
