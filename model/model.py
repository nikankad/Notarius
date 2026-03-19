# imports
import torch.nn as nn
import torch


class TSCConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, use_relu=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv1d(in_channel, in_channel, kernel_size,
                      padding, groups=in_channel),
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
        ]
        if use_relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QuartNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, R=5):
        super().__init__()

        self.net = nn.Sequential(
            # On our Rth block,
            *[TSCConv(in_channel if i == 0 else out_channel, out_channel, kernel_size, use_relu=(i != R-1)) for i in range(R)]
        )
        # residual
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))


class QuartzNetBxR(nn.Module):
    def __init__(self, n_mels=64, n_classes=29, R=5, B=5):
        super().__init__()
        assert B % 5 == 0, "B Must be a mutiple of 5"
        self.net = nn.Sequential(
            # c1
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # B1-5
            # B1
            *[QuartNetBlock(256, 256, kernel_size=33, R=R)
              for _ in range(B//5)],
            # B2
            *[QuartNetBlock(256, 256, kernel_size=39, R=R)
              for _ in range(B//5)],
            # B3
            # first: transitions 256→512
            QuartNetBlock(256, 512, kernel_size=51, R=R),
            # rest: stays at 512            #B4
            *[QuartNetBlock(512, 512, kernel_size=51, R=R)
              for _ in range(B//5 - 1)],
            *[QuartNetBlock(512, 512, kernel_size=63, R=R)
              for _ in range(B//5)],
            # B5
            *[QuartNetBlock(512, 512, kernel_size=75, R=R)
              for _ in range(B//5)],
            # C2
            nn.Conv1d(512, 512, kernel_size=87, padding=43),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # C3
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # C4
            nn.Conv1d(1024, n_classes, dilation=2, kernel_size=1),
        )

    def forward(self, x):
        x = self.net(x)  # (batch, classes, time)
        x = x.permute(2, 0, 1)      # (time, batch, n_classes)
        return x.log_softmax(dim=2)
