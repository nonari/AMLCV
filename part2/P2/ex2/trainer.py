import importlib
from pytorch_lightning import Trainer, Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import DataModule
from models import GenericNet


class MetricTracker(Callback):
    def __init__(self):
        self.validation_scores = []
        self.best = 0

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        score = pl_module.val_epoch_results[-1]
        if score > self.best:
            self.best = score
            version = trainer.logger.version
            save_dir = trainer.logger.save_dir
            name = trainer.logger.name
            trainer.save_checkpoint(f'{save_dir}/{name}/version_{version}/best.ckpt')

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f'Callback: {pl_module.val_epoch_results}')
        self.validation_scores.append(pl_module.val_epoch_results[-1])


def train(config_name, resume=False, version=0, batch=None, check_path=None):
    CONFIG = importlib.import_module(f'configs.{config_name}').CONFIG

    if check_path is not None:
        CONFIG['log']['root'] = check_path
    print(CONFIG['log']['root'])
    checkpoint = pl.callbacks.ModelCheckpoint(**CONFIG['checkpoint'])

    if batch is not None:
        CONFIG['dataset']['batch_train'] = batch
    dataset = DataModule(CONFIG['dataset'])

    model = GenericNet(CONFIG['model'])

    logconf = CONFIG['log']

    metric_tracker = MetricTracker()

    logger = TensorBoardLogger(logconf['root'], name=logconf['name'])

    checkpoint_route = None
    if resume:
        checkpoint_route = f'{logconf["root"]}/{logconf["name"]}/version_{version}/checkpoints/best.ckpt'

    trainer = Trainer(accelerator='gpu', logger=logger, callbacks=[metric_tracker, checkpoint],
                      resume_from_checkpoint=checkpoint_route, max_epochs=20)

    trainer.fit(model, datamodule=dataset)


# train('resnet18_simple', resume=False, version=1)
