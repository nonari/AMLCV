import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolveTwice(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolve = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2))
        self.conv = ConvolveTwice(in_channels, out_channels)

    def forward(self, x_r, x_l):
        x_r = self.up(x_r)
        x_r = self.conv_up(x_r)
        diff_y = x_l.size()[2] - x_r.size()[2]
        diff_x = x_l.size()[3] - x_r.size()[3]

        x_r = F.pad(x_r, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x_l, x_r], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, classes, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)
