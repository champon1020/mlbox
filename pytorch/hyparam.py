"""
Search best hyperparameters with Optuna.

"""

import mlflow
import optuna
import torch.optim as optim

from datasets.dataset import make_datasets
from main import parse_args
from models.model import make_model
from train import Training

# tmp
BATCH_SIZE = 64
EPOCHS = 200


def objective(trial: optuna.trial.Trial):
    """
    Optuna objective function.

    """
    # Parse CLI arguments.
    args = parse_args()

    # Prepare some instances for training.
    train_ds, valid_ds, _ = make_datasets(args.train_path, args.valid_path, None)

    # Suggestions.
    # >>> TODO >>>

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSProp", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", low=1e-6, hight=1e-1, log=True)
    n_layer = trial.suggest_int("n_layer", 32, 1024, step=32)
    norm = trial.suggest_categorical("norm", ["BatchNorm", "LayerNorm"])

    # <<< TODO <<<

    model = make_model(n_layer=n_layer, norm=norm)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    training = Training(train_ds, valid_ds, model, optimizer, BATCH_SIZE, EPOCHS)

    for epoch in range(EPOCHS):
        # Training phase.
        training.model.train()
        training.process(training.train_loader)

        # Validation phase.
        training.model.eval()
        _, accuracy = training.process(training.valid_loader)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def mlflow_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Callback function for mlflow.

    """
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=study.study_name):
        # >>> TODO >>>

        mlflow.log_params(trial.params)
        mlflow.log_metrics({"accuracy": trial_value})

        # <<< TODO <<<


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[mlflow_callback])

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    best_trial = study.best_trial

    print("Study statistics:")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of completed trials: ", len(completed_trials))
    print("")
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params:")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
