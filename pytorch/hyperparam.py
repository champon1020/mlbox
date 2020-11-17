"""
Search best hyperparameters with Optuna.

"""
import mlflow
import numpy as np
import optuna
import torch.optim as optim

from configs import HyperparamConfig
from datasets import Dataset
from models import make_model
from train import Training, parse_cli_args


def objective(trial: optuna.trial.Trial):
    """
    Optuna objective function.

    """
    # Parse CLI arguments.
    args = parse_cli_args()

    config = HyperparamConfig()

    # Prepare dataset class.
    train_ds = Dataset(args.train_path)
    valid_ds = Dataset(args.valid_path)

    # Suggestions.
    # >>> TODO >>>

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSProp", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", low=1e-6, hight=1e-1, log=True)
    n_layer = trial.suggest_int("n_layer", 32, 1024, step=32)

    # <<< TODO <<<

    model = make_model(n_layer=n_layer)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    training = Training(
        train_ds,
        valid_ds,
        model,
        optimizer,
        config.batch_size,
        config.epochs,
    )

    accuracy_list = []

    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        for epoch in range(config.epochs):
            # Training phase.
            training.model.train()
            train_loss, train_accuracy = training.train_epoch()
            mlflow.log_metrics(
                {"train_loss": train_loss, "train_accuracy": train_accuracy},
                step=epoch + 1,
            )

            # Validation phase.
            training.model.eval()
            valid_loss, valid_accuracy = training.validate()
            mlflow.log_metrics(
                {"valid_loss": valid_loss, "valid_accuracy": valid_accuracy},
                step=epoch + 1,
            )

            accuracy_list.append(valid_accuracy.item())

            trial.report(valid_accuracy.item(), epoch)

            if trial.should_prune():
                print("Pruned with epoch {}".format(epoch))
                raise optuna.exceptions.TrialPruned()

            print(
                "Epoch {}: TrainLoss: {:.5f}, ValidLoss: {:.5f}, ValidAcc: {:.5f}".format(
                    epoch + 1, train_loss, valid_loss, valid_accuracy
                )
            )

        accuracy_list.sort()
        accuracy = np.mean(accuracy_list[-10:])
        mlflow.log_metrics({"top10_avg_accuracy": accuracy})

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=100)

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
