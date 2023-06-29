from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


class OptunaPruningLightningCallback(PyTorchLightningPruningCallback, L.Callback):
    # see git issue: https://github.com/optuna/optuna/issues/4689
    # tldr: Optuna not compatible with lightning v2 yet
    def __init__(self, trial: optuna.Trial, monitor: str, *args, **kwargs) -> None:
        super().__init__(trial=trial, monitor=monitor, *args, **kwargs)


class OptunaHyperparameterLogger:
    def __init__(
        self,
        logdir: Path,
        logfile: str = "sampled_hparams.json",
    ):
        self.logdir = logdir
        self.logfile = logfile

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        trial_id = trial.number
        logfile = self.logdir.joinpath(f"trial_{trial_id}").joinpath(self.logfile)
        with logfile.open("w") as fp:
            json.dump(trial.params, fp, indent=4)
            fp.write("\n")
