import torch
import sys
sys.path.insert(0, '..')
from unet import UNet


CONFIG = {
    'log': {
        'root': '.tb_logs',
        'name': 'unet_simple'
    },
    'dataset': {
        'batch_train': 4,
        'batch_test': 4,
        'validation': True,
        'augment': False
    },
    'checkpoint': {
        'dirpath': None,
        'save_last': True,
        'every_n_epochs': 10,
    },
    'model': {
        'class': UNet,
        'params': {
            'n_classes': 11
        },
        'optimizer': {
            'class': torch.optim.SGD,
            'params': {
                'lr': 0.001,
                'momentum': 0.9
            },
            'lr_scheduler': {
                'class': torch.optim.lr_scheduler.CyclicLR,
                'params': {
                    'base_lr': 0.0001,
                    'max_lr': 0.01
                },
            }
        },
    }
}