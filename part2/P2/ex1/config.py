import torch
import torchvision

CONFIG = {
    'log': {
        'root': '.tb_logs',
        'name': 'resnet18'
            },
    'dataset': {
        'batch_train': 4,
        'batch_test': 4,
        'validation': True,
    },
    'model': {
        'class': torchvision.models.ResNet,
        'params': {
            'block': torchvision.models.resnet.Bottleneck,
            'layers': [3, 4, 6, 3],
            'num_classes': 10
        },
        'optimizer': {
            'class': torch.optim.SGD,
            'params': {
                'lr': 0.001,
                'momentum': 0.9
            },
            'lr_scheduler': {
                'class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                'params': {
                    'T_0': 10,
                },
            }
        },
    }
}