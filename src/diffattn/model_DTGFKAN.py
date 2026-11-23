import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter


# ========================= 原始ParamNonHarmonicEncoding的多层扩展 =========================
class MultiLayerParamNonHarmonicEncoding(nn.Module):
    """
    多层非谐波傅里叶级数编码器
    - 直接基于model_mk.py中的ParamNonHarmonicEncoding实现
    - 当num_layers=1时完全等同于原始ParamNonHarmonicEncoding
    """

    def __init__(self, k, hidden_dim=128, num_freq=None, Omega=50.0, delta_min=0.25,
                 weight_penalty=1e-4, num_layers=1, residual=True):
        """
        Args:
            k: 外层循环次数（与原始实现一致）
            hidden_dim: 输入维度 ≈ 1 + 2*num_freq
            num_freq: 使用的频率个数（默认 hidden_dim//2）
            Omega: 最大频率（带宽约束）
            delta_min: 最小间隔（保证频率分离，避免病态）
            weight_penalty: L2 正则项系数
            num_layers: 编码器层数
            residual: 是否使用残差连接
        """
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim 必须为偶数"

        self.k = k
        self.Omega = Omega
        self.delta_min = delta_min
        self.weight_penalty = weight_penalty
        self.num_layers = int(num_layers)
        self.residual = residual

        self.num_freq = num_freq if num_freq is not None else hidden_dim // 2

        # 为每一层创建独立的参数
        # 正确的方式：将参数作为类属性
        self.freq_deltas_layers = nn.ParameterList([
            nn.Parameter(torch.rand(self.num_freq) * 0.5) for _ in range(self.num_layers)
        ])

        self.freq_bias_layers = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_layers)
        ])

        # 创建多层多通道的线性层
        self.readouts_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(1 + 2 * self.num_freq, 1) for _ in range(k)])
            for _ in range(self.num_layers)
        ])

        # 注册常量缓冲区
        self.register_buffer("one_bias", torch.tensor(1.0))

    def _construct_freqs(self, layer_idx):
        """
        构造严格递增频率（与原始ParamNonHarmonicEncoding完全相同）
        """
        deltas = F.softplus(self.freq_deltas_layers[layer_idx]) + self.delta_min
        freqs = torch.cumsum(deltas, dim=0) + self.freq_bias_layers[layer_idx]
        freqs = self.Omega * torch.tanh(freqs / self.Omega)
        return freqs

    def get_and_print_frequencies(self, epoch=None):
        """
        获取并打印当前所有层的频率序列
        Args:
            epoch: 当前训练轮数（可选）
        """
        for layer_idx in range(self.num_layers):
            freqs = self._construct_freqs(layer_idx)

            # 计算均匀密度指标
            uniformity_metric = self.check_uniform_density(freqs).item()

            # 计算频率间隔
            intervals = freqs[1:] - freqs[:-1]
            avg_interval = torch.mean(intervals).item()
            min_interval = torch.min(intervals).item()
            max_interval = torch.max(intervals).item()

            # 打印频率信息
            epoch_info = f"Epoch {epoch}: " if epoch is not None else ""
            print(f"{epoch_info}Layer {layer_idx} 频率统计:")
            print(f"  均匀密度指标: {uniformity_metric:.4f} (越接近0越均匀)")
            print(f"  平均间隔: {avg_interval:.4f}, 最小间隔: {min_interval:.4f}, 最大间隔: {max_interval:.4f}")
            print(f"  频率范围: [{freqs[0].item():.4f}, {freqs[-1].item():.4f}]")

            # 如果频率数量不多，则打印完整序列
            if self.num_freq <= 20:
                print("  完整频率序列:", [f"{f.item():.4f}" for f in freqs])
            else:
                # 否则只打印前5个和后5个
                print("  前5个频率:", [f"{f.item():.4f}" for f in freqs[:5]])
                print("  后5个频率:", [f"{f.item():.4f}" for f in freqs[-5:]])
            print()

            # 可选：返回一个包含频率序列的字典，用于更高级的分析或可视化
            freq_stats = {
                "layer": layer_idx,
                "freqs": freqs.detach().cpu().numpy(),
                "uniformity": uniformity_metric,
                "avg_interval": avg_interval,
                "min_interval": min_interval,
                "max_interval": max_interval
            }

            return freq_stats

    def check_uniform_density(self, freqs, window_size=10):
        """
        检查频率序列是否满足均匀密度约束
        Args:
            freqs: 频率序列
            window_size: 窗口大小
        Returns:
            均匀密度偏差度量（越小越好）
        """
        # 计算理想情况下的间隔
        ideal_interval = self.Omega / self.num_freq

        # 计算实际间隔的标准差（用于衡量均匀性）
        actual_intervals = freqs[1:] - freqs[:-1]
        uniformity_metric = torch.std(actual_intervals) / ideal_interval

        return uniformity_metric

    def _weight_penalty(self):
        """
        权重L2正则化
        """
        reg = 0.0
        for layer_idx in range(self.num_layers):
            for lin in self.readouts_layers[layer_idx]:
                reg = reg + (lin.weight ** 2).sum()
        return reg * self.weight_penalty

    def regularization_loss(self):
        """
        返回正则化损失（保持与原始接口一致）
        """
        return self._weight_penalty()

    def forward(self, e):
        """
        前向传播
        Args:
            e: 特征值 [N]
        Returns:
            与原始ParamNonHarmonicEncoding兼容的输出列表 [N,1] * k
        """
        e = e.view(-1)
        ee = e.unsqueeze(1)

        # 初始化每个通道的输出
        outputs = [None] * self.k

        # 逐层处理
        for layer_idx in range(self.num_layers):
            freqs = self._construct_freqs(layer_idx)
            norm_scale = (self.num_freq ** 0.5)

            # 为每个通道计算结果
            layer_outputs = []
            for i in range(self.k):
                # 与原始实现完全相同的处理流程
                ei = ee.pow(i + 1)
                phase = ei * freqs.unsqueeze(0)

                feat = torch.cat([
                    torch.ones_like(ee) * self.one_bias,  # 常数通道
                    torch.sin(phase) / norm_scale,
                    torch.cos(phase) / norm_scale,
                ], dim=1)

                out = self.readouts_layers[layer_idx][i](feat)
                layer_outputs.append(out)

            # 处理多层的连接方式
            if layer_idx == 0:
                # 第一层直接赋值
                outputs = layer_outputs
            else:
                # 后续层添加到前面的结果（可选是否使用残差连接）
                for i in range(self.k):
                    if self.residual:
                        outputs[i] = outputs[i] + layer_outputs[i]
                    else:
                        outputs[i] = layer_outputs[i]

        return outputs


# ========================= 基于原始代码的GrokFormer_KAN =========================
class NoFoDifformer(nn.Module):
    """
    GrokFormer与多层非谐波傅里叶KAN结合
    - 使用与model_mk完全相同的结构
    - 仅将ParamNonHarmonicEncoding替换为多层版本
    """

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, dim=32, nheads=1, k=10,
                 num_layers=1, num_freq=None, Omega=50.0, delta_min=0.25, weight_penalty=1e-4, residual=True,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super().__init__()

        # GrokFormer_mk的结构
        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # 替换为多层版本
        self.eig_encoder = MultiLayerParamNonHarmonicEncoding(
            k=k,
            hidden_dim=dim,
            num_freq=num_freq,
            Omega=Omega,
            delta_min=delta_min,
            weight_penalty=weight_penalty,
            num_layers=num_layers,
            residual=residual
        )

        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        self.k = k
        self.alpha = nn.Linear(self.k, 1, bias=False)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        # 确定特征维度
        feat_dim = nclass if norm == 'none' else hidden_dim

        # 添加门控机制
        self.gate_norm = nn.LayerNorm(feat_dim * 3)
        self.gate_linear = nn.Linear(feat_dim * 3, 3)  # 三个门控权重

        if norm == 'none':
            self.mha_norm = nn.LayerNorm(nclass)
            self.ffn_norm = nn.LayerNorm(nclass)
            self.mha = Diff_MultiHeadAttention_Optimized(nclass, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(nclass, nclass, nclass)
        else:
            self.mha_norm = nn.LayerNorm(hidden_dim)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.mha = Diff_MultiHeadAttention_Optimized(hidden_dim, nheads, tran_dropout)
            self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, nclass)

    def transformer_encoder(self, h, h_fur, layer):
        mha_h = self.mha_norm(h)
        mha_h = self.mha(mha_h, mha_h, mha_h, layer)
        mha_h_dropout = self.mha_dropout(mha_h)

        # 使用门控机制替代直接相加
        concat_features = torch.cat([h, mha_h_dropout, h_fur], dim=-1)
        gate_weights = self.gate_norm(concat_features)
        gate_weights = self.gate_linear(gate_weights)
        gate_weights = F.softmax(gate_weights, dim=-1).unsqueeze(-1)

        # 使用门控权重进行加权融合
        mha_h_ = h * gate_weights[:, 0] + mha_h_dropout * gate_weights[:, 1] + h_fur * gate_weights[:, 2]

        ffn_h = self.ffn_norm(mha_h_)
        ffn_h = self.ffn(ffn_h)
        encoder_h = mha_h_ + self.ffn_dropout(ffn_h)
        return encoder_h

    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig = self.eig_encoder(e)
        new_e = torch.cat(eig, dim=1)
        new_e = self.alpha(new_e)

        for conv in range(self.nlayer):
            utx = ut @ h
            h_encoder = u @ (new_e * utx)
            h_encoder = self.prop_dropout(h_encoder)
            h = self.transformer_encoder(h, h_encoder, conv)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h


# ========================= FeedForward =========================
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


# ========================= MultiHeadAttention (Diff Version) =========================
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Diff_MultiHeadAttention_Optimized(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Diff_MultiHeadAttention_Optimized, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.hidden_size = hidden_size

        self.linear_q1 = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_k1 = nn.Linear(hidden_size, num_heads * hidden_size)

        self.linear_q2 = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_k2 = nn.Linear(hidden_size, num_heads * hidden_size)

        self.linear_v = nn.Linear(hidden_size, num_heads * hidden_size)

        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.layer_norm = nn.LayerNorm(num_heads * hidden_size)
        self.output_layer = nn.Linear(num_heads * hidden_size, hidden_size)

        self.lambda_init = lambda_init_fn(0)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.hidden_size).normal_(mean=0, std=0.1))

    def forward(self, q, k, v, layer, attn_bias=None):
        orig_q_size = q.size()

        q1 = self.linear_q1(q)
        k1 = self.linear_k1(k)
        q2 = self.linear_q2(q)
        k2 = self.linear_k2(k)
        v = self.linear_v(v)

        # 优化：先计算 KV
        k1v = k1.transpose(0, 1).matmul(v)
        k2v = k2.transpose(0, 1).matmul(v)

        k1v = self.att_dropout(k1v)
        k2v = self.att_dropout(k2v)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        self.lambda_init = lambda_init_fn(layer)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        x1 = torch.matmul(q1, k1v)
        x2 = torch.matmul(q2, k2v) * lambda_full
        x = x1 - x2

        x = self.layer_norm(x)
        x = x * (1 - self.lambda_init)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x