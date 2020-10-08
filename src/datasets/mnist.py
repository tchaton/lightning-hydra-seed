import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from src.core.base_dataset import BaseDataset

class MNISTTDataset(BaseDataset):

    NAME = "mnist"

    def __init__(
        self,
        *args,
        name: str = 'mnist',
        val_split: float = 0.3,
        **kwargs,
    ):
        assert 0 < val_split and val_split < 1
        self.NAME = name
        self.val_split = val_split
        super().__init__(*args, **kwargs)


    @property
    def num_features(self):
        return 28 * 28

    @property
    def num_classes(self):
        return 10

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    def prepare_data(self):
        """
        Saves MNIST files to data_root
        """
        train_dataset = MNIST(
            self.data_root, train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = MNIST(
            self.data_root,
            train=False,
            download=True,
            transform=self.train_transform,
        )

        train_length = int(len(train_dataset) * (1 - self.val_split))
        self.train_dataset, self.val_dataset = random_split(
            train_dataset,
            [train_length, len(train_dataset) - train_length],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.create_dataloaders()
