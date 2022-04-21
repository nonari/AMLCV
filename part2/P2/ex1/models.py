import torchvision
import resnet
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torchmetrics import Accuracy


class GenericNet(LightningModule):
    # Override
    def __init__(self, config):
        super().__init__()
        self.val_epoch_results = []
        self.config = config
        self.model = None
        self.loss = None
        self.accuracy = Accuracy()
        self.init_model()
        self.epoch = 0

    def init_model(self):
        self.model = self.config['class'](**self.config['params'])
        old_weight = self.model.conv1.weight.sum(dim=1, keepdim=True)
        self.model.conv1 = nn.Conv2d(1, self.model.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.model.conv1.weight.data = old_weight
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        y = self.model(x)
        return y

    def do_step(self, batch, phase):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss(logits, y)

        y_hat = torch.argmax(logits, dim=1)
        acc = self.accuracy(y, y_hat)

        # self.logger.log_metrics({f'loss_{phase}': loss, f'acc_{phase}': acc}, step=self.trainer.global_step)

        output = {'loss': loss, 'acc': acc}

        return output

    # Override
    def training_step(self, batch, batch_idx):
        output = self.do_step(batch, 'train')
        return output

    # Override
    def validation_step(self, batch, batch_idx):
        output = self.do_step(batch, 'validation')
        return output

    # Override
    def test_step(self, batch, batch_idx):
        output = self.do_step(batch, 'test')
        return output

    def do_epoch_end(self, outputs, phase):
        losses = [output['loss'] for output in outputs]
        accs = [output['acc'] for output in outputs]

        loss = torch.mean(torch.stack(losses))
        acc = torch.mean(torch.stack(accs))

        self.log(f'epoch_loss_{phase}', loss, on_epoch=True)
        self.log(f'epoch_acc_{phase}', acc, on_epoch=True)

        if phase == 'validation':
            self.val_epoch_results.append(acc.item())

        print(f'\n\n\nEpoch {self.epoch} {phase} loss: {loss}\n')

    # Override
    def training_epoch_end(self, training_step_outputs):
        self.do_epoch_end(training_step_outputs, 'train')
        self.epoch += 1

    # Override
    def validation_epoch_end(self, validation_step_outputs):
        self.do_epoch_end(validation_step_outputs, 'validation')

    # Override
    def test_epoch_end(self, test_step_outputs):
        self.do_epoch_end(test_step_outputs, 'test')

    # Override
    def configure_optimizers(self):
        opt_conf = self.config['optimizer']
        optimizer = opt_conf['class'](self.model.parameters(), **opt_conf['params'])

        sch_conf = opt_conf['lr_scheduler']
        scheduler = sch_conf['class'](optimizer, **sch_conf['params'])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
