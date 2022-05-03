import importlib
from pytorch_lightning import Trainer, Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import MNISTDataModule
from models import GenericNet
import torch



def test(config_name, check_path=None, batch=4, version=0):
    CONFIG = importlib.import_module(f'configs.{config_name}').CONFIG

    if check_path is not None:
        CONFIG['log']['root'] = check_path
    if batch is not None:
        CONFIG['dataset']['batch_train'] = batch
    dataset = MNISTDataModule(CONFIG['dataset'])

    dir_conf = CONFIG['log']
    root = dir_conf['root']
    name = dir_conf['name']

    model_config = CONFIG['model']

    device = torch.device('cpu') if torch.cuda.is_available() else torch.devide('cpu')

    model = model_config['class'](**model_config['params'])

    saved_state = torch.load(f'{root}/{name}/version_{version}/best.ckpt')

    model.load_state_dict(dict([(n[6:], p) for n, p in saved_state['state_dict'].items()]), strict=True)

    # model.load_state_dict(saved_state['state_dict'])
    model.eval()
    model.to(device)

    iterations = 0
    val_accuracy = 0
    datapoints = 0
    for images, targets in dataset.test_dataloader():
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            logits = model(images)

            iterations += 1

            preds = torch.argmax(logits, dim=1).flatten()
            datapoints += preds.shape[0]
            val_accuracy += (preds == targets.flatten()).sum().item()

    print()

    val_accuracy /= datapoints
    print(val_accuracy)


test('resnet18_pre_aug', version=0)
