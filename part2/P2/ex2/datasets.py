from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np
from numpy import random


sample_transform = transforms.Compose([
    transforms.Pad(96),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])


class SegmentationMNIST(Dataset):
    def __init__(self, root: str, train: bool, image_size=(128, 128), transform=None, target_transform=None,
                 ndigits=(5, 8), max_iou=0.1):
        """
        Args:
        - root: the route for the MNIST dataset. It will be downloaded if it does not exist
        - train: True for training set, False for test set
        - image_size: tuple with the dataset image size
        - transform: the transforms to be applied to the input image
        - target_transform: the transforms to be applied to the label image
        - ndigits: tuple with the mininum and maximum number of digits per image
        - max_iou: maximum IOU between digit bounding boxes
        """

        self.transform = transform
        self.target_transform = target_transform
        self.image_template = torch.zeros((1, *image_size), dtype=torch.float32)
        self.target_tempalte = torch.zeros((1, *image_size), dtype=torch.uint8)
        self.max_iou = 0.2

        mnist_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.5, 1.5), shear=30),
            transforms.Resize(48),
            transforms.ToTensor()
        ])

        self.mnist = datasets.MNIST(root, train, download=True, transform=mnist_transform)
        self.index = random.permutation(len(self.mnist))

        # Compute the number of digits in each image, and the total number of images
        self.num_digits = []
        remaining = len(self.mnist)
        while remaining > ndigits[1] + ndigits[0]:  # The remaining will be from min to max, i.e. one image
            this_num = random.randint(ndigits[0], ndigits[1])
            self.num_digits.append(this_num)
            remaining -= this_num
        self.num_digits.append(remaining)

        self.num_digits = np.array(self.num_digits)
        self.start_digit = self.num_digits.cumsum() - self.num_digits[0]

    def __len__(self):
        return len(self.num_digits)

    def __getitem__(self, idx):
        sample = self.image_template.detach().clone()
        target = self.image_template.detach().clone()

        for i in range(self.num_digits[idx]):
            if self.start_digit[idx] + i >= len(self.mnist):
                break
            digit, cls = self.mnist[self.start_digit[idx] + i]
            mask = digit > 0

            valid = False
            while not valid:
                y = random.randint(0, sample.size(1) - digit.size(1))
                x = random.randint(0, sample.size(2) - digit.size(2))
                valid = (mask * sample[:, y:y + digit.size(1),
                                x:x + digit.size(2)] > 0).sum() < self.max_iou * mask.sum()

            sample[:, y:y + digit.size(1), x:x + digit.size(2)][mask] = digit[mask]
            target[:, y:y + digit.size(1), x:x + digit.size(2)][mask] = cls + 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DataModule(LightningDataModule):
    # Override
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.validation = SegmentationMNIST(root="./data", train=False, transform=sample_transform)
        self.rem = self.validation.__len__() % 2
        self.half = int(self.validation.__len__() / 2)

    # Override
    def train_dataloader(self):
        dataset = SegmentationMNIST(root="./data", train=True, transform=sample_transform)
        return DataLoader(dataset, batch_size=self.config['batch_train'])

    # Override
    def val_dataloader(self):
        _, val_set = torch.utils.data.random_split(self.validation,
                                                   [self.half+self.rem, self.half], torch.Generator().manual_seed(42))
        return DataLoader(val_set, batch_size=self.config['batch_train'])

    # Override
    def test_dataloader(self):
        test_set, _ = torch.utils.data.random_split(self.validation,
                                                    [self.half+self.rem, self.half], torch.Generator().manual_seed(42))
        return DataLoader(test_set, batch_size=self.config['batch_train'])
