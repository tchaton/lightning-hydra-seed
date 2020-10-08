import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.core.base_model import BaseModel


def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                nn.BatchNorm1d(channels[i], momentum=bn_momentum),
                nn.Dropout(p=0.2),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )


class SimpleMLP(BaseModel):
    def __init__(self, *args, num_features: int = None, num_classes: int = None, **kwargs):
        super(SimpleMLP, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.mlp = MLP([num_features, 128, 128, num_classes])

    def forward(self, x):
        return self.mlp(x.view(x.size(0), -1))
