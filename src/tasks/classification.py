import os
import os.path as osp
import numpy as np
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import metrics
import pytorch_lightning as pl
from collections import namedtuple

class CategoricalClassificationStepsMixin:
    def __init__(self, *args, **kwargs):

        if len(kwargs) > 0:
            self._prog_bar = kwargs.get("prog_bar", False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss.item(), self._prog_bar)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, self._prog_bar)

