import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAC(nn.Module):
    def __init__(self, channel, reduction=16, num_groups=4):
        super(GAC, self).__init__()
        self.num_groups = num_groups
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."
        self.group_channels = channel // num_groups
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.group_channels, out_channels=self.group_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.group_channels // reduction, out_channels=self.group_channels, kernel_size=1)
        )
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x_grouped = x.view(batch_size, self.num_groups, self.group_channels, height, width)
        x_pooled = x_grouped.view(batch_size * self.num_groups, self.group_channels, height, width)
        x_h_avg = F.adaptive_avg_pool2d(x_pooled, (height, 1));
        x_h_max = F.adaptive_max_pool2d(x_pooled, (height, 1))
        x_w_avg = F.adaptive_avg_pool2d(x_pooled, (1, width));
        x_w_max = F.adaptive_max_pool2d(x_pooled, (1, width))
        y_h_avg = self.shared_conv(x_h_avg);
        y_h_max = self.shared_conv(x_h_max)
        y_w_avg = self.shared_conv(x_w_avg);
        y_w_max = self.shared_conv(x_w_max)
        y_h_avg = y_h_avg.view(batch_size, self.num_groups, self.group_channels, height, 1)
        y_h_max = y_h_max.view(batch_size, self.num_groups, self.group_channels, height, 1)
        y_w_avg = y_w_avg.view(batch_size, self.num_groups, self.group_channels, 1, width)
        y_w_max = y_w_max.view(batch_size, self.num_groups, self.group_channels, 1, width)
        att_h = self.sigmoid_h(y_h_avg + y_h_max)
        att_w = self.sigmoid_w(y_w_avg + y_w_max)
        out = x_grouped * att_h * att_w
        return out.view(batch_size, channel, height, width)


class SCE(nn.Module):
    def __init__(self, in_channels=1024, embed_dim=256):
        super(SCE, self).__init__()
        self.conv_foreground = GAC(channel=in_channels)
        self.conv_background = GAC(channel=in_channels)
        self.dim_reducer_fg = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.dim_reducer_bg = nn.Conv2d(in_channels, embed_dim, kernel_size=1)


    def forward(self, feature, mask_foreground, return_mode=None):
        mask_foreground = F.interpolate(mask_foreground, size=(feature.shape[2], feature.shape[3]), mode='bilinear',
                                        align_corners=False)
        mask_background = 1 - mask_foreground
        f_foreground = mask_foreground * feature
        f_background = mask_background * feature
        f_foreground_conv = self.conv_foreground(f_foreground)
        f_background_conv = self.conv_background(f_background)

        f_foreground_reduced = self.dim_reducer_fg(f_foreground_conv)
        f_background_reduced = self.dim_reducer_bg(f_background_conv)

        f_foreground_reduced = f_foreground_reduced
        f_background_reduced = f_background_reduced
        seq_fg = f_foreground_reduced.flatten(2).permute(2, 0, 1)
        seq_bg = f_background_reduced.flatten(2).permute(2, 0, 1)

        final_sequence = torch.cat([seq_fg, seq_bg], dim=0)
        if return_mode == 'viz':
            return {
                'final_sequence': final_sequence,
                'fg_feature_reduced': f_foreground_reduced,
                'bg_feature_reduced': f_background_reduced
            }
        return final_sequence