import os
import logging
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_class
from pytorch_lightning.utilities.model_utils import is_overridden
from src.core.base_dataset import BaseDataset
from src.core.base_model import BaseModel

def custom_instantiate(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    t_cls = get_class(cfg["_target_"])
    return t_cls(**cfg)

def attach_step_and_epoch_functions(model, datamodule):
    datamodule.forward = model.forward
    datamodule.log = model.log
    for attr in dir(datamodule):
        if sum([token in attr for token in ["_step", "_epoch_end", "dataloader"]]) > 0:
            if not is_overridden(attr, model):
                setattr(model, attr, getattr(datamodule, attr))

def check_task_possible(task, cfg):
    assert task in [c._target_ for c in cfg], f"Provided {task} isn't supported by {cfg}"

def initialize_task(cfg: DictConfig):
    # Extract task mixin
    task_target = cfg.task._target_
    check_task_possible(task_target, cfg.model.authorized_tasks) 
    check_task_possible(task_target, cfg.dataset.authorized_tasks) 

    cfg.dataset.task_target = task_target
    data_module: BaseDataset = custom_instantiate(cfg.dataset)
    model: BaseModel = instantiate(cfg.model, **data_module.hyper_parameters, optimizers=cfg.optimizers.optimizers)

    attach_step_and_epoch_functions(model, data_module)

    data_module.prepare_data()

    return model, data_module