#imports
import torch.nn as nn
import torch

class IBConv(nn.Module):
    """Inverted Bottleneck 1D Time-Channel Separable Convolution Module"""
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, expand=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        mid_channel = in_channel * expand

        # [MODIFIED] Added BN+ReLU after expand and after depthwise.
        # original code had: expand → depthwise → compress → BN → ReLU
        # Correct pattern: expand → BN → ReLU → depthwise → BN → ReLU → compress → BN (NO ReLU)
        self.net = nn.Sequential(
            # 1) Pointwise expand: C → C*t
            nn.Conv1d(in_channel, mid_channel, kernel_size=1, bias=False),   # [MODIFIED] bias=False
            nn.BatchNorm1d(mid_channel),                                      # [ADDED] BN after expand
            nn.ReLU(),                                                        # [ADDED] ReLU after expand

            # 2) Depthwise conv in expanded space
            nn.Conv1d(mid_channel, mid_channel, kernel_size, stride, padding,
                      groups=mid_channel, bias=False),                        # [MODIFIED] bias=False
            nn.BatchNorm1d(mid_channel),                                      # [ADDED] BN after depthwise
            nn.ReLU(),                                                        # [ADDED] ReLU after depthwise

            # 3) Pointwise compress: C*t → C_out
            nn.Conv1d(mid_channel, out_channel, kernel_size=1, bias=False),   # [MODIFIED] bias=False
            nn.BatchNorm1d(out_channel),                                      # BN after compress (same as yours)
            # [MODIFIED] NO ReLU here — this is the "linear bottleneck" from MobileNetV2.
            # ReLU in the narrow compressed space destroys information.
            # Non-linearity already exists inside the module (after expand and depthwise).
        )

        # [ADDED] Per-module residual skip when channels match.
        self.use_residual = (in_channel == out_channel and stride == 1)

    def forward(self, x):
        out = self.net(x)
        # [ADDED] per-module residual
        if self.use_residual:
            out = out + x
        return out

class IBBlock(nn.Module):
    """Block of R inverted bottleneck modules with block-level residual"""
    def __init__(self, in_channel, out_channel, kernel_size, R=3, expand=2):
        super().__init__()

        # First module handles channel change (in_channel → out_channel)
        self.layer1 = IBConv(in_channel, out_channel, kernel_size, expand=expand)
        # Remaining R-1 modules: same channels, each has internal per-module residual
        self.layers = nn.ModuleList([
            IBConv(out_channel, out_channel, kernel_size, expand=expand) for _ in range(R-1)
        ])

        # Block-level residual (same as original QuartzNet)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),  # [MODIFIED] bias=False
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        out = self.layer1(x)
        for layer in self.layers:
            out = layer(out)
        # Block-level residual + ReLU (same as QuartzNet)
        out = out + self.residual(x)
        return torch.relu(out)

class IBNet(nn.Module):
    # [MODIFIED] expand=4 → expand=2.
    # With expand=4 and 512 channels: mid=2048, each pointwise = 512×2048 = 1M params.
    # That gave you 41.9M total — 6× larger than QuartzNet 5x5.
    # With expand=2: mid=1024, each pointwise = 512×1024 = 524K. Much more reasonable.
    #
    # [MODIFIED] R=5 → R=3.
    # Each IB module has 3 convolutions (expand+DW+compress) vs QuartzNet's 2 (DW+PW).
    # So R=3 IB modules ≈ R=5 QuartzNet modules in terms of total conv layers per block.
    #
    # [MODIFIED] channels 256/512 → configurable via C parameter.
    # With C=256/512, expand=2, R=3: 14.1M — too large.
    # Recommended configs for paper experiments:
    #   C=172 → 6.7M  (param-matched to QuartzNet 5x5)
    #   C=192 → 8.2M  (between QuartzNet 5x5 and 10x5)
    #   C=256 → 14.1M (param-matched to QuartzNet 10x5~15x5)
    def __init__(self, n_mels=64, n_classes=29, R=3, expand=2, C=192):
        super().__init__()
        C2 = C * 2   # channels for B3-B5 and C2
        C3 = C2 * 2  # channels for C3
        self.net = nn.Sequential(
            #c1 (same as QuartzNet)
            nn.Conv1d(n_mels, C, kernel_size=33, stride=2, padding=16, bias=False),  # [MODIFIED] bias=False
            nn.BatchNorm1d(C),
            nn.ReLU(),
            #B1
            IBBlock(C, C, kernel_size=33, R=R, expand=expand),
            #B2
            IBBlock(C, C, kernel_size=39, R=R, expand=expand),
            #B3
            IBBlock(C, C2, kernel_size=51, R=R, expand=expand),
            #B4
            IBBlock(C2, C2, kernel_size=63, R=R, expand=expand),
            #B5
            IBBlock(C2, C2, kernel_size=75, R=R, expand=expand),
            #C2 (single IB module, no block residual needed)
            IBConv(C2, C2, kernel_size=87, expand=expand),
            #C3
            nn.Conv1d(C2, C3, kernel_size=1, bias=False),  # [MODIFIED] bias=False
            nn.BatchNorm1d(C3),
            nn.ReLU(),
            #C4 (output layer, bias=True is fine here)
            nn.Conv1d(C3, n_classes, dilation=2, kernel_size=1),
        )

    def forward(self, x):
        x = x.squeeze(1) # now: (batch, n_mels, time)
        return self.net(x)
