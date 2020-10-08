import os
import inspect
import os.path as osp
import numpy as np
from omegaconf import OmegaConf
from functools import partial
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from src.core.base_task import BaseTasksMixin


class BaseDataset(BaseTasksMixin, LightningDataModule):

    NAME = ...

    def __init__(self, *args, **kwargs):

        BaseTasksMixin.__init__(self, *args, **kwargs)
        self.__instantiate_transform(kwargs)
        self.__clean_kwargs(kwargs)
        LightningDataModule.__init__(self)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.seed = 42
        self._num_workers = 2
        self._shuffle = True
        self._drop_last = False
        self._pin_memory = True
        self._follow_batch = []

        self._hyper_parameters = {}

        self.data_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME)

    @staticmethod
    def del_attr(kwargs, name):
        try:
            del kwargs[name]
        except:
            pass

    def __clean_kwargs(self, kwargs):
        LightningDataModuleArgs = inspect.getargspec(LightningDataModule.__init__).args
        keys = list(kwargs.keys())
        for key in keys:
            if key not in LightningDataModuleArgs:
                self.del_attr(kwargs, key)

    def __instantiate_transform(self, kwargs):
        self.pre_transform = None
        self.transform = None
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        for k in [k for k in kwargs]:
            if "transform" in k and kwargs.get(k) is not None:
                transforms = []
                for t in kwargs.get(k):
                    if "activate" in t:
                        if t["activate"] is False:
                            continue
                        del t["activate"]
                    transforms.append(instantiate(t))
                transform = T.Compose(transforms)
                setattr(self, f"{k}", transform)
                del kwargs[k]

    @property
    def num_features(self):
        pass

    @property
    def num_classes(self):
        pass

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    def create_dataloaders(self):
        for stage in ["train", "val", "test"]:
            stage_dataset = f"{stage}_dataset"
            if hasattr(self, stage_dataset):
                stage_dataloader = f"{stage}_dataloader"
                dataset = getattr(self, stage_dataset)
                partial_func = partial(self._dataloader, dataset=dataset)
                partial_func.__code__ = self._dataloader
                setattr(self, stage_dataloader, partial_func)

    
    def _dataloader(self, batch_size=32, transforms=None, dataset=None):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader