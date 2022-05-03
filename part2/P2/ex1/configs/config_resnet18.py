import torch
import sys
sys.path.insert(0, '..')
from resnet import ResNetGray


CONFIG = {
    'log': {
        'root': '.tb_logs',
        'name': 'resnet18'
    },
    'dataset': {
        'batch_train': 4,
        'batch_test': 4,
        'validation': True,
        'augment': True
    },
    'checkpoint': {
        'dirpath': None,
        'save_last': True,
        'every_n_epochs': 10,
    },
    'model': {
        'class': ResNetGray,
        'params': {
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