from pytorch_lightning import Trainer, Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import MNISTDataModule
from models import GenericNet
from config import CONFIG


class MetricTracker(Callback):
    def __init__(self):
        self.validation_scores = []

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f'Callback: {pl_module.val_epoch_results}')
        self.validation_scores.append(pl_module.val_epoch_results[-1])


checkpoint = pl.callbacks.ModelCheckpoint(CONFIG['checkpoint'])

dataset = MNISTDataModule(CONFIG['dataset'])

model = GenericNet(CONFIG['model'])

logconf = CONFIG['log']

metric_tracker = MetricTracker()

logger = TensorBoardLogger(logconf['root'], name=logconf['name'])
trainer = Trainer(accelerator='cpu', logger=logger, callbacks=[metric_tracker, checkpoint])

trainer.fit(model, datamodule=dataset)
