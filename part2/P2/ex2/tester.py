import importlib
from pytorch_lightning import Trainer, Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import DataModule
from models import GenericNet
import torch
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def test(config_name, check_path=None, batch=4, version=0):
    CONFIG = importlib.import_module(f'configs.{config_name}').CONFIG

    if check_path is not None:
        CONFIG['log']['root'] = check_path
    if batch is not None:
        CONFIG['dataset']['batch_train'] = batch
    dataset = DataModule(CONFIG['dataset'])

    dir_conf = CONFIG['log']
    root = dir_conf['root']
    name = dir_conf['name']

    model_config = CONFIG['model']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model_config['class'](**model_config['params'])

    saved_state = torch.load(f'{root}/{name}/version_{version}/best.ckpt')

    model.load_state_dict(dict([(n[6:], p) for n, p in saved_state['state_dict'].items()]), strict=True)

    # model.load_state_dict(saved_state['state_dict'])
    model.eval()
    model.to(device)

    iterations = 0
    val_accuracy = 0
    datapoints = 0
    softmax = torch.nn.Softmax()
    all_targets = np.zeros((0, 1, 128, 128))
    all_preds = np.zeros((0, 11, 128, 128))
    for images, targets in dataset.test_dataloader():
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            logits = model(images)
            logits = logits[:, :, 2:-2, 2:-2]
            iterations += 1

            preds = torch.argmax(logits, dim=1).flatten()
            datapoints += preds.shape[0]
            val_accuracy += (preds.flatten() == targets.flatten()).sum().item()
            all_preds = np.concatenate((all_preds, softmax(logits).detach().numpy()))
            all_targets = np.concatenate((all_targets, targets.detach().numpy()))

    print()

    val_accuracy /= datapoints
    print(val_accuracy)

    for c in range(11):
        fpr, tpr, t = roc_curve(np.int0(all_targets.flatten() == c), all_preds[:, c].flatten())
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.axline((0, 0), (1, 1), color='blue', linestyle='dashed')
        plt.title(f'Class: {c if c > 0 else "Background"}')
        plt.grid('on')
        plt.show()

    cm = confusion_matrix(all_targets.flatten(), np.argmax(all_preds, axis=1).flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


# test('unet_simple', version=0)
