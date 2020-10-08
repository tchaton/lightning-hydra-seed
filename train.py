import os
import logging
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate

os.environ["HYDRA_FULL_ERROR"] = "1"

import torch
import pytorch_lightning as pl

from src.config import *
from src.datasets import *
from src.core import *
from src import initialize_task
from src.utils.loggers import initialize_loggers
from src.utils.jit import check_jittable

log = logging.getLogger(__name__)


def train(cfg):
    log.info(OmegaConf.to_yaml(OmegaConf.to_container(cfg, resolve=True)))
    OmegaConf.set_struct(cfg, False)

    model, data_module = initialize_task(cfg)

    loggers: List[pl.callbacks.Callback] = initialize_loggers(
        cfg
    )  # , **model.config, **data_module.config

    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    trainer: pl.Trainer = instantiate(cfg.trainer, gpus=gpus, logger=loggers)

    if cfg.jit:
        check_jittable(model, data_module)

    trainer.fit(model, data_module)
    log.info("Training complete.")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
