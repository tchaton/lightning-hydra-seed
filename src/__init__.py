import os
import logging
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_class
from pytorch_lightning.utilities.model_utils import is_overridden
from src.core.base_dataset import BaseDataset
from src.core.base_model import BaseModel


def custom_instantiate(dataset_cfg, task_cfg):
    dataset_dict = OmegaConf.to_container(dataset_cfg, resolve=True)
    task_dict = OmegaConf.to_container(task_cfg, resolve=True)
    dataset_dict["task_config"] = task_dict
    t_cls = get_class(dataset_dict["_target_"])
    return t_cls(**dataset_dict)


def attach_step_and_epoch_functions(model, datamodule):
    datamodule.forward = model.forward
    datamodule.log = model.log
    for attr in dir(datamodule):
        if sum([token in attr for token in ["_step", "_epoch_end", "dataloader"]]) > 0:
            if not is_overridden(attr, model):
                setattr(model, attr, getattr(datamodule, attr))


def check_task_possible(task, cfg):
    assert task in [
        c._target_ for c in cfg
    ], f"Provided {task} isn't supported by {cfg}"


def initialize_task(cfg: DictConfig):
    # Extract task mixin
    check_task_possible(cfg.task._target_, cfg.model.authorized_tasks)
    check_task_possible(cfg.task._target_, cfg.dataset.authorized_tasks)

    data_module: BaseDataset = custom_instantiate(cfg.dataset, cfg.task)
    model: BaseModel = instantiate(
        cfg.model, **data_module.hyper_parameters, optimizers=cfg.optimizers.optimizers
    )

    attach_step_and_epoch_functions(model, data_module)

    data_module.prepare_data()

    return model, data_module
