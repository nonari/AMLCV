from torch.utils.data import DataLoader
import datasets
# import models
from models import GenericNet
import torch.optim as optim
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


trainset = datasets.mnist(train=True)

batch_size = 4
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


# net = models.resNet('resnet18')
model = GenericNet('resnet18')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

base_lr = 1e-2
min_lr = 1e-4
period = 10
patience = 20
momentum = 0.9
weight_decay = 1e-3


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=base_lr,
    momentum=momentum,
    weight_decay=weight_decay
)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=period, eta_min=min_lr)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')