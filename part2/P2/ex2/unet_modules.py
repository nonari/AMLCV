import torch
import torch.nn as nn


class ConvolveTwice(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvolveTwice(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(2, 2), stride=(2, 2))
        self.conv = ConvolveTwice(in_channels, out_channels)

    def forward(self, x_r, x_l):
        x_r = self.up(x_r)

        diff_y = x_l.shape[2] - x_r.shape[2]
        diff_x = x_l.shape[3] - x_r.shape[3]

        x_l = x_l[:, :, diff_x // 2:-diff_x // 2, diff_y // 2:-diff_y // 2]
        x = torch.cat([x_l, x_r], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, classes, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)
