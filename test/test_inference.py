import os
import sys
import warnings

warnings.simplefilter("ignore")
from omegaconf import OmegaConf
import pytest
import functools

sys.path.append("..")
from hydra.experimental import compose, initialize

from src.config import cs
from train import train

DIR_PATH = os.path.dirname(os.path.dirname(__file__))


def getcwd():
    return os.path.join(DIR_PATH, "outputs")


current_cwd = os.getcwd()
os.getcwd = getcwd


def run(*outer_args, **outer_kwargs):
    def runner_func(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if runs[func.__name__]:
                return func(*args, **kwargs)
            return 0

        return func_wrapper

    return runner_func


local_runs = {"test_simple_mlp_mnist": True}

workflow_runs = {"test_simple_mlp_mnist": True}

if "runner" in current_cwd:
    runs = workflow_runs
else:
    runs = local_runs

################ CORA TESTS ################
@pytest.mark.parametrize("task", ["categorical_classification"])
@pytest.mark.parametrize("model", ["simple_mlp"])
@pytest.mark.parametrize("dataset", ["mnist"])
@pytest.mark.parametrize("jit", ["False", "True"])
@run()
def test_simple_mlp_mnist(task, model, dataset, jit):
    cmd_line = "task={} model={} dataset={} loggers=thomas-chaton log=false jit={}"
    with initialize(config_path="../conf", job_name="test_app"):
        print({"model": model, "dataset": dataset, "jit": jit})
        cfg = compose(
            config_name="config",
            overrides=cmd_line.format(task, model, dataset, jit).split(" "),
        )
        train(cfg)
