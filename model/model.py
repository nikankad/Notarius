#imports
import torch.nn as nn
import torch

class TSCConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, relu=True, expand=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        mid_channel = in_channel * expand
        layers = [
            nn.Conv1d(in_channel, mid_channel, kernel_size=1),          # expand
            nn.Conv1d(mid_channel, mid_channel, kernel_size, stride, padding, groups=mid_channel),  # depthwise
            nn.Conv1d(mid_channel, out_channel, kernel_size=1),          # project
            nn.BatchNorm1d(out_channel),
        ]
        if relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class QuartNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, R=5, expand=1):
        super().__init__()

        self.layer1 = TSCConv(in_channel, out_channel, kernel_size, expand=expand)
        self.layers = nn.ModuleList([
            TSCConv(out_channel, out_channel, kernel_size, expand=expand) for _ in range(R-2)
        ])
        self.layer_final = TSCConv(out_channel, out_channel, kernel_size, relu=False, expand=expand)

        #residual
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        out = self.layer1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.layer_final(out)
        out = out + self.residual(x)
        return torch.relu(out)

class Notarius(nn.Module):
    def __init__(self, n_mels=64, n_classes=29, B=5, R=5, expand=4):
        super().__init__()
        self.net = nn.Sequential(
            #c1
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #B1
            QuartNetBlock(256, 256, kernel_size=33, R=R, expand=expand),
            #B2
            QuartNetBlock(256, 256, kernel_size=39, R=R, expand=expand),
            #B3
            QuartNetBlock(256, 512, kernel_size=51, R=R, expand=expand),
            #B4
            QuartNetBlock(512, 512, kernel_size=63, R=R, expand=expand),
            #B5
            QuartNetBlock(512, 512, kernel_size=75, R=R, expand=expand),
            #C2
            TSCConv(512, 512, kernel_size=87, expand=expand),
            #C3
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #C4
            nn.Conv1d(1024, n_classes, dilation=2, kernel_size=1),
        )

    def forward(self, x):
        x = x.squeeze(1) # now: (batch, n_mels, time)
        return self.net(x)
