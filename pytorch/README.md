# Machine Learning Template Project

## Description
This project uses awesome libraries [mlflow](https://github.com/mlflow/mlflow) and [optuna](https://github.com/optuna/optuna).

And the libraries make our machine learning experiments more confortable.

## Usage
Run training.
```
mlflow experiments create -n train_runs

mlflow run -e train --experiment-name train_runs \
    -P train-path=/path/to/train/data \
    -P valid-path=/path/to/valid/data
```

Tune hyperparameters.
```
mlflow experiments create -n hyperparam_runs

mlflow run -e hyperparam --experiment-name hyperparam_runs \
    -P train-path=/path/to/train/data \
    -P valid-path=/path/to/valid/data

```

Run evaluation.
```
python evaluate.py \
    --test-path /path/to/test/data \
    --ckpt-path /path/to/checkpoint
```

## Project Structure
```
pytorch
├── configs
│   ├── __init__.py
│   ├── default.py           -> Abstract dataclass.
│   ├── evaluate_config.py   -> Evaluation configuration dataclass.
│   ├── hyperparam_config.py -> Hyperparameters tuning configuration dataclass.
│   └── train_config.py      -> Training configuration dataclass.
│
├── datasets
│   ├── __init__.py
│   └── dataset.py           -> Dataset class.
│
├── miscs
│   ├── __init__.py
│   ├── metrics.py           -> Metrics functions.
│   └── output.py            -> Output functions.
│
├── models
│   ├── __init__.py
│   └── model.py             -> Model class.
│
├── MLproject                -> Mlflow project file.
├── evaluate.py              -> Evaluation entrypoint.
├── hyperparam.py            -> Hyperparameter tuning entrypoint.
└── train.py                 -> Training entrypoint.
```
