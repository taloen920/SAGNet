import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGlobalFilter(nn.Module):


    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.complex_weight = None
        self.weight_generator = nn.Linear(2, dim * 2)  # (h,w)
    def forward(self, x):
        B, C, H, W = x.shape
        if self.complex_weight is None or self.complex_weight.size(0) != H or self.complex_weight.size(1) != W // 2 + 1:
            h_idx = torch.linspace(-1, 1, H, device=x.device)
            w_idx = torch.linspace(-1, 1, W // 2 + 1, device=x.device)
            grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')
            coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
            weights = self.weight_generator(coords).view(H, W // 2 + 1, self.dim, 2)
            self.complex_weight = nn.Parameter(weights, requires_grad=True)

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class CRAFT(nn.Module):

    def __init__(self, dim, ratio=4, num_classes=2):
        super(CRAFT, self).__init__()
        inter_channels = dim // ratio

        self.Global = AdaptiveGlobalFilter(inter_channels)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.conv5a = nn.Sequential(
            nn.Conv2d(dim, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.Global(feat1) + feat1
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        return sa_output

