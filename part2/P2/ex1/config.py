import torch

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
        'class': 'resnet18',
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