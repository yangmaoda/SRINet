import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class Learnable2DPositionalEncoding(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim)
        )

    def forward(self, h: int, w: int, device=None) -> torch.Tensor:
        ys = torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device)
        xs = torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
        pos = self.net(coords)
        return pos.unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredSpatialAttention_Seq(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=False)

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor, value_gate: Optional[torch.Tensor] = None):
        """
        Optional `value_gate` controls value scaling.
        value_gate shape: (1, B, 1)
        """
        SeqLen, B, D = x.shape
        H = self.mha.num_heads
        attn_mask = head_mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * H, SeqLen, SeqLen)

        q, k, v = x, x, x

        if value_gate is not None:
            # Broadcast from (1, B, 1) to (SeqLen, B, D).
            v = v * value_gate

        attn_output, _ = self.mha(q, k, v, attn_mask=attn_mask)
        return attn_output


class GlobalGater_Seq(nn.Module):
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None: hidden_dim = max(32, d_model // 4)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2), nn.Sigmoid()
        )

    def forward(self, seq_s: torch.Tensor, seq_b: torch.Tensor):
        # seq_s, seq_b: [SeqLen_half, B, D]
        f_s_global = seq_s.mean(dim=0)  # [B, D]
        f_b_global = seq_b.mean(dim=0)  # [B, D]

        g_in = torch.cat([f_s_global, f_b_global, (f_s_global - f_b_global).abs()], dim=1)
        gates = self.mlp(g_in)

        g_homo = gates[:, 0].unsqueeze(0).unsqueeze(-1)  # Shape: (1, B, 1) to broadcast
        g_hetero = gates[:, 1].unsqueeze(0).unsqueeze(-1)  # Shape: (1, B, 1)

        return g_homo, g_hetero

class CIE(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 ffn_ratio: int = 4):
        super(CIE, self).__init__()
        self.D = d_model
        self.H = num_heads

        self.gater = GlobalGater_Seq(d_model)
        self.attention_homo = StructuredSpatialAttention_Seq(d_model, num_heads // 2)
        self.attention_hetero = StructuredSpatialAttention_Seq(d_model, num_heads // 2)
        self.norm_in = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_ratio, dropout)
        self.head_masks = {}

    def create_masks(self, total_seq_len, device):
        if str(total_seq_len) in self.head_masks:
            return self.head_masks[str(total_seq_len)]
        H_half = self.H // 2
        seq_len_half = total_seq_len // 2
        mask_homo = torch.full((H_half, total_seq_len, total_seq_len), float("-inf"), device=device)
        mask_homo[:, :seq_len_half, :seq_len_half] = 0.0
        mask_homo[:, seq_len_half:, seq_len_half:] = 0.0
        mask_hetero = torch.full((H_half, total_seq_len, total_seq_len), float("-inf"), device=device)
        mask_hetero[:, :seq_len_half, seq_len_half:] = 0.0
        mask_hetero[:, seq_len_half:, :seq_len_half] = 0.0
        self.head_masks[str(total_seq_len)] = (mask_homo, mask_hetero)
        return mask_homo, mask_hetero

    def forward(self, sequence: torch.Tensor):
        total_seq_len = sequence.shape[0]
        seq_len_half = total_seq_len // 2
        seq_s_res = sequence[:seq_len_half]
        seq_b_res = sequence[seq_len_half:]

        g_homo, g_hetero = self.gater(seq_s_res, seq_b_res)

        sequence_norm = self.norm_in(sequence)

        mask_homo, mask_hetero = self.create_masks(total_seq_len, sequence.device)

        out_homo = self.attention_homo(sequence_norm, mask_homo, value_gate=g_homo)
        out_hetero = self.attention_hetero(sequence_norm, mask_hetero, value_gate=g_hetero)

        interacted_sequence = sequence + out_homo + out_hetero

        out_sequence = interacted_sequence + self.ffn(interacted_sequence)

        return out_sequence


def flatten_nf(F_interact: torch.Tensor) -> torch.Tensor:
    B, Nf, HW, D = F_interact.shape
    return F_interact.permute(0, 2, 1, 3).contiguous().view(B, HW, Nf * D)