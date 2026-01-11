import torch
from torch import nn


class RVQ(nn.Module):
    def __init__(
        self,
        K: int,
        V_list,
        code_dim: int,
        ema_decay: float = 0.99,
        usage_balance_w: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.K = K
        self.V_list = list(V_list)
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.usage_balance_w = usage_balance_w
        self.device = device

        self.codebooks = nn.ParameterList()
        self.ema_counts = nn.ParameterList()
        self.ema_sums = nn.ParameterList()
        for v in self.V_list:
            cb = nn.Parameter(torch.randn(v, code_dim, device=device) * 0.02, requires_grad=False)
            counts = nn.Parameter(torch.ones(v, device=device), requires_grad=False)
            sums = nn.Parameter(cb.detach().clone(), requires_grad=False)
            self.codebooks.append(cb)
            self.ema_counts.append(counts)
            self.ema_sums.append(sums)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, return_inputs: bool = False):
        residual = x
        codes = []
        quantized = torch.zeros_like(x)
        head_inputs = []
        for k in range(self.K):
            cb = self.codebooks[k]
            head_input = residual
            d = (
                residual.unsqueeze(1) - cb.unsqueeze(0)
            ).pow(2).sum(dim=-1)
            if self.usage_balance_w > 0:
                counts = self.ema_counts[k]
                usage = counts / (counts.mean() + 1e-6)
                d = d + self.usage_balance_w * usage.unsqueeze(0)
            code = d.argmin(dim=1)
            q = cb[code]
            codes.append(code)
            quantized = quantized + q
            residual = residual - q
            head_inputs.append(head_input)
        codes = torch.stack(codes, dim=1)
        if return_inputs:
            return codes, quantized, residual, head_inputs
        return codes, quantized, residual

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        codes, _, _, head_inputs = self.encode(x, return_inputs=True)
        for k in range(self.K):
            v = self.V_list[k]
            codes_k = codes[:, k]
            inputs_k = head_inputs[k]
            counts = torch.bincount(codes_k, minlength=v).float()
            sums = torch.zeros(v, self.code_dim, device=x.device)
            sums.index_add_(0, codes_k, inputs_k)

            self.ema_counts[k].mul_(self.ema_decay).add_(counts * (1.0 - self.ema_decay))
            self.ema_sums[k].mul_(self.ema_decay).add_(sums * (1.0 - self.ema_decay))

            denom = self.ema_counts[k].unsqueeze(1).clamp(min=1e-6)
            self.codebooks[k].data.copy_(self.ema_sums[k] / denom)

            dead = self.ema_counts[k] < 1.0
            if dead.any():
                rand_idx = torch.randint(0, x.size(0), (dead.sum().item(),), device=x.device)
                self.codebooks[k].data[dead] = inputs_k[rand_idx]
                self.ema_counts[k].data[dead] = 1.0
                self.ema_sums[k].data[dead] = inputs_k[rand_idx]

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor):
        out = torch.zeros(codes.size(0), self.code_dim, device=codes.device)
        for k in range(self.K):
            cb = self.codebooks[k]
            out = out + cb[codes[:, k]]
        return out


def load_rvq(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model = RVQ(
        K=ckpt["config"]["K"],
        V_list=ckpt["config"]["V_list"],
        code_dim=ckpt["config"]["code_dim"],
        ema_decay=ckpt["config"]["ema_decay"],
        usage_balance_w=ckpt["config"]["usage_balance_w"],
        device=device,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
