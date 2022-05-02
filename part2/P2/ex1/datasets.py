import torchvision
import torch
from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


normal_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
     transforms.Resize((56, 56), interpolation=transforms.InterpolationMode.NEAREST),
     transforms.Pad(98)])

augmentation_transform = [
    transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine([-10, 10], translate=[0.1, 0.1], scale=[0.9, 1.1]),
        transforms.Normalize(mean=0.1307, std=0.3081)
    ])
]


def mnist(train=True, augmentation=False):
    root = 'data/'

    transform = augmentation_transform if augmentation else normal_transform
    dataset = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)

    return dataset


class MNISTDataModule(LightningDataModule):
    # Override
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.validation = config['validation']

    # Override
    def train_dataloader(self):
        dataset = mnist(train=True)
        t, _ = torch.utils.data.random_split(dataset, [16, 59984])
        return DataLoader(t, batch_size=self.config['batch_train'])

    # Override
    def val_dataloader(self):
        dataset = mnist(train=False)
        # val, val_set = torch.utils.data.random_split(dataset, [16, 9984])
        _, val_set = torch.utils.data.random_split(dataset, [9992, 8])
        return DataLoader(val_set, batch_size=self.config['batch_train'])

    # Override
    def test_dataloader(self):
        dataset = mnist(train=False)
        if not self.validation:
            return DataLoader(dataset)
        test_set, _ = torch.utils.data.random_split(dataset, [8, 9992])
        return DataLoader(test_set, batch_size=self.config['batch_train'])
