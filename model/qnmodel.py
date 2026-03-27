#imports
import torch.nn as nn 
import torch
class TSCConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, relu=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv1d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel),
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
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
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel)
        )
    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))

class QuartzNetBxR(nn.Module):
    def __init__(self, n_mels=64, n_classes=29, B=5, R=5):
        super().__init__()
        self.net = nn.Sequential(
            #c1
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #B1-5
            #B1
            QuartNetBlock(256, 256, kernel_size=33, R=5),
            #B2
            QuartNetBlock(256, 256, kernel_size=39, R=5),
            #B3
            QuartNetBlock(256, 512, kernel_size=51, R=5),
            #B4
            QuartNetBlock(512, 512, kernel_size=63, R=5),
            #B5
            QuartNetBlock(512, 512, kernel_size=75, R=5),
            #C2
            TSCConv(512, 512, kernel_size=87),
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


