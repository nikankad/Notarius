#imports
import torch.nn as nn 
import torch
class TSCConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, relu=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv1d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=False),
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel),
        ]
        if relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class QuartNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, R=5):
        super().__init__()
    
        self.net = nn.Sequential(
            TSCConv(in_channel, out_channel, kernel_size),
            *[TSCConv(out_channel, out_channel, kernel_size) for _ in range(R-2)],
            TSCConv(out_channel, out_channel, kernel_size, relu=False),
        )
        #residual 
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )
    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))

class QuartzNetBxR(nn.Module):
    def __init__(self, n_mels=64, n_classes=29, B=5, R=5):
        super().__init__()

        # S = how many times each block is repeated consecutively
        # B=5 -> S=1, B=10 -> S=2, B=15 -> S=3
        S = B // 5

        # Block definitions: (in_channel, out_channel, kernel_size)
        block_defs = [
            (256, 256, 33),   # B1
            (256, 256, 39),   # B2
            (256, 512, 51),   # B3
            (512, 512, 63),   # B4
            (512, 512, 75),   # B5
        ]

        blocks = []

        # C1 - initial conv
        blocks.extend([
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        ])

        # Each block Bi is repeated S times consecutively
        # e.g. 10x5: B1-B1-B2-B2-B3-B3-B4-B4-B5-B5
        for in_c, out_c, ks in block_defs:
            for s in range(S):
                if s == 0:
                    blocks.append(QuartNetBlock(in_c, out_c, kernel_size=ks, R=R))
                else:
                    blocks.append(QuartNetBlock(out_c, out_c, kernel_size=ks, R=R))

        # C2, C3, C4 - final layers
        blocks.extend([
            TSCConv(512, 512, kernel_size=87),
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, n_classes, dilation=2, kernel_size=1),
        ])

        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        x = x.squeeze(1) # now: (batch, n_mels, time)
        return self.net(x)


