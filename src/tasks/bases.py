import os
import os.path as osp
from enum import Enum
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from pytorch_lightning import metrics
import pytorch_lightning as pl
from collections import namedtuple


class GANMode(Enum):
    Encoder = "gen"
    Discriminator = "dis"


class VGAEMode(Enum):
    Encoder = "enc"
    Discriminator = "dis"


class BaseStepsMixin:
    def inference_step(self, batch, batch_nb, stage, generative_mode=None):
        print("Hello")

    def step(self, batch, batch_nb, stage, generative_mode=None):
        print("Hello")

    def prepare_batch(self, batch, batch_nb, stage, sampling):
        print("Hello")

    def validation_step(self, batch, batch_nb):
        print("Hello")

    def test_step(self, batch, batch_nb):
        print("Hello")

    def training_step(self, batch, batch_nb, sampling=None):
        print("Hello")

    def _test_step(self,batch,batch_nb,stage=None):
        print("Hello")
