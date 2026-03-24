import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from einops import rearrange

from SCE import SCE
from CIE import CIE, flatten_nf


class CDSA(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(CDSA, self).__init__()

        # Channel attention branch.
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        # Spatial attention branch.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        x = x * x_channel_att
        x = self.channel_shuffle(x, groups=4)
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


class CrossAttentionBlock(nn.Module):
    """
    A complete cross-attention block following Transformer practices.
    Includes cross-attention, layer norm, residual connections, and FFN.
    """

    def __init__(self, embed_dim, num_heads, ffn_ratio=4, dropout=0.1, activation="gelu"):
        super(CrossAttentionBlock, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):

        query_residual = query

        q_norm = self.norm1(query)
        k_norm = self.norm_kv(key)
        v_norm = self.norm_kv(value)

        attn_output, attn_output_weights = self.multihead_attn(q_norm, k_norm, v_norm)

        query = query_residual + self.dropout1(attn_output)

        query_residual = query

        q_norm = self.norm2(query)

        ffn_output = self.ffn(q_norm)

        query = query_residual + self.dropout2(ffn_output)

        return query, attn_output_weights


class CustomResNet50_v2(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.5, drop_rate_resnet=0.2,
                 embed_dim=256, num_heads=8, ffn_ratio=4):
        super(CustomResNet50_v2, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stage1 = nn.Sequential(*list(self.resnet.children())[:5], nn.Dropout(drop_rate_resnet))
        self.stage2 = nn.Sequential(*list(self.resnet.children())[5], nn.Dropout(drop_rate_resnet))
        self.stage3 = nn.Sequential(*list(self.resnet.children())[6], nn.Dropout(drop_rate_resnet))
        self.cdsa = CDSA(in_channels=1024)
        self.conv_cdsa = nn.Conv2d(1024, embed_dim, kernel_size=1)
        self.SCE = SCE(in_channels=1024, embed_dim=embed_dim)
        self.cie = CIE(d_model=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=drop_rate)
        self.cross_attention = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.fc_score = nn.Sequential(nn.Linear(embed_dim, 128), nn.SiLU(), nn.Dropout(drop_rate), nn.Linear(128, 64),
                                      nn.SiLU(), nn.Dropout(drop_rate), nn.Linear(64, 1))
        self.fc_weight = nn.Sequential(nn.Linear(embed_dim, 128), nn.SiLU(), nn.Dropout(drop_rate), nn.Linear(128, 64),
                                       nn.SiLU(), nn.Dropout(drop_rate), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x, mask, return_mode=None):
        B = x.size(0)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        _, _, H, W = x3.shape

        x_cdsa_raw = self.cdsa(x3)
        x_cdsa = self.conv_cdsa(x_cdsa_raw)
        x_cdsa_flatten = x_cdsa.flatten(2).permute(0, 2, 1)

        if return_mode == 'viz':
            sce_outputs = self.SCE(x3, mask, return_mode='viz')
            sequence = sce_outputs['final_sequence']
        else:
            sequence = self.SCE(x3, mask)

        interacted_sequence = self.cie(sequence)
        x_cie_query = interacted_sequence.permute(1, 0, 2)

        x_cross, attn_cross_map = self.cross_attention(query=x_cie_query,
                                                       key=x_cdsa_flatten,
                                                       value=x_cdsa_flatten)

        if return_mode == 'viz':
            w = self.fc_weight(x_cross)
            viz_outputs = {
                "H": H, "W": W,

                "x1_map": x1,
                "x2_map": x2,
                "x3_map": x3,

                "cdsa_map": x_cdsa_raw,

                "sce_fg_feature": sce_outputs['fg_feature_reduced'],
                "sce_bg_feature": sce_outputs['bg_feature_reduced'],

                "cie_output_seq": x_cie_query,

                "cross_attention_output_seq": x_cross,
                "cross_attention_map": attn_cross_map,

                "weights": w,
            }
            return viz_outputs

        f = self.fc_score(x_cross)
        w = self.fc_weight(x_cross)
        fw = f * w
        sum_fw = torch.sum(fw, dim=1)
        sum_w = torch.sum(w, dim=1)
        scores = sum_fw / (sum_w + 1e-6)

        return scores.squeeze(-1)