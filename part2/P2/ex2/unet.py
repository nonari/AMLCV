from unet_modules import *


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()

        self.conv1 = ConvolveTwice(1, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv2 = ConvolveTwice(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv3 = ConvolveTwice(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv4 = ConvolveTwice(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv5 = ConvolveTwice(512, 1024)
        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        self.out = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        x3 = self.conv3(x3)
        x4 = self.down3(x3)
        x4 = self.conv4(x4)
        x5 = self.down4(x4)
        x5 = self.conv5(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits
