import torch
import torchvision


class ResNetGray(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnetGray = torchvision.models.resnet.resnet18(pretrained=pretrained)
        old_weight = self.resnetGray.conv1.weight.sum(dim=1, keepdim=True)
        self.resnetGray.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnetGray.conv1.weight.data = old_weight
        self.resnetGray.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnetGray(x)
