import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierMassEncoder(nn.Module):
    def __init__(self, num_freq=64, max_mz=1000.0):
        super().__init__()
        # 线性分布频率，也可按论文做高低频分组
        freqs = torch.linspace(1.0, num_freq, num_freq) / max_mz
        self.register_buffer("freqs", freqs)

    def forward(self, mz):  # mz: [B, N]
        # [B, N, 2*num_freq]
        mz = mz.unsqueeze(-1)  # [B,N,1]
        angles = 2 * torch.pi * mz * self.freqs  # [B,N,F]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class DreaMSEncoder(nn.Module):
    def __init__(
        self,
        output_size=4096,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_freq=64,
        dropout=0.1,
        max_peaks=100,
        use_int_preds=False,
    ):
        super().__init__()
        self.max_peaks = max_peaks
        self.use_int_preds = use_int_preds

        self.mz_encoder = FourierMassEncoder(num_freq=num_freq)
        self.int_encoder = nn.Linear(1, hidden_size)

        # 输入维度：2*num_freq (mz) + hidden_size (intensity)
        in_dim = 2 * num_freq + hidden_size
        self.proj = nn.Linear(in_dim, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.fp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        # 兼容 aux
        if use_int_preds:
            self.intermediate_head = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        # batch 需要提供:
        #   batch["mz"]: [B,N], batch["intens"]: [B,N], batch["num_peaks"] or batch["mask"]
        mz = batch["mz"]
        intens = batch["intens"]

        # 构造 mask（padding 为 0）
        if "mask" in batch:
            mask = batch["mask"]  # [B,N] True=valid
        else:
            num_peaks = batch["num_peaks"]
            max_len = mz.size(1)
            mask = torch.arange(max_len, device=mz.device)[None, :] < num_peaks[:, None]

        # 加 precursor token (m0, intensity=1.1)
        precursor_mz = batch.get("precursor_mz")  # [B]
        if precursor_mz is not None:
            precursor_mz = precursor_mz.unsqueeze(1)  # [B,1]
        else:
            # 若无，可设置为 0
            precursor_mz = torch.zeros((mz.size(0), 1), device=mz.device)

        precursor_int = torch.ones_like(precursor_mz) * 1.1

        mz = torch.cat([precursor_mz, mz], dim=1)
        intens = torch.cat([precursor_int, intens], dim=1)

        # 更新 mask
        precursor_mask = torch.ones((mz.size(0), 1), device=mz.device, dtype=mask.dtype)
        mask = torch.cat([precursor_mask, mask], dim=1)

        # 编码
        mz_feat = self.mz_encoder(mz)                  # [B,N,2F]
        int_feat = self.int_encoder(intens.unsqueeze(-1))  # [B,N,H]

        x = torch.cat([mz_feat, int_feat], dim=-1)
        x = self.proj(x)

        # Transformer 需要 padding mask：True=padding
        padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # 使用 precursor token 作为全局表征
        h0 = x[:, 0, :]  # [B,H]

        output = self.fp_head(h0)  # [B,4096]

        aux = {"h0": h0}
        if self.use_int_preds:
            aux["int_preds"] = [self.intermediate_head(h0)]

        return output, aux