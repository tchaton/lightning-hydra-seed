# Pytorch Lightning seed project with hydra

[![codecov](https://codecov.io/gh/tchaton/lightning-hydra-seed/branch/master/graph/badge.svg)](https://codecov.io/gh/tchaton/lightning-hydra-seed) [![Actions Status](https://github.com/tchaton/lightning-hydra-seed/workflows/unittest/badge.svg)](https://github.com/tchaton/lightning-hydra-seed/actions)

### Setup

```
pyenv install 3.7.8
pyenv local 3.7.8
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install poetry
python -m poetry install
```

### PRINCIPAL CMD

```python
python train.py model={{MODEL}} dataset={{DATASET}} loggers={{LOGGERS}} log={{LOG}} notes={{NOTES}} name={{NAME}} jit={{JIT}}
```

- `LOGGERS` str: Configuration file to log to Wandb, currently using mine as `thomas-chaton`
- `LOG` bool: Wheter to log training to wandb
- `NOTES` str: A note associated to the training
- `NAME` str: Training name appearing on Wandb.
- `LOG` bool: Wheter to make model jittable.

### Current Demo Command

```python
python train.py task=categorical_classification model=simple_mlp dataset=mnist loggers=thomas-chaton log=False
```

```python
python train.py task=categorical_classification model=simple_mlp dataset=mnist loggers=thomas-chaton log=False notes="My Simple MLP with Adam" notes="My Simple MLP with Adam" name="My Simple MLP with Adam" jit=False
```
