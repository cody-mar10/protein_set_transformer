from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

import optuna

from pst.utils.cli import _KWARG_TYPE

MODEL_COMPLEXITY = Literal["small", "medium", "large", "xl", "xxl"]


@dataclass
class ModelComplexityValues:
    num_heads: int
    n_enc_layers: int


MODEL_COMPLEXITY_MAP = {
    "small": ModelComplexityValues(num_heads=4, n_enc_layers=5),
    "medium": ModelComplexityValues(num_heads=8, n_enc_layers=10),
    "large": ModelComplexityValues(num_heads=16, n_enc_layers=15),
    "xl": ModelComplexityValues(num_heads=32, n_enc_layers=20),
    "xxl": ModelComplexityValues(num_heads=32, n_enc_layers=30),
}


class HyperparameterRegistryMixin:
    def __init__(self, trial: optuna.Trial):
        self._hparams: _KWARG_TYPE = dict()
        self._trial = trial

    def _register_model_hparams(self):
        complexity_choices = cast(tuple[str, ...], get_args(MODEL_COMPLEXITY))
        complexity = self._trial.suggest_categorical(
            "complexity", choices=complexity_choices
        )
        complexity_values = MODEL_COMPLEXITY_MAP[complexity]
        self._hparams["num_heads"] = complexity_values.num_heads
        self._hparams["n_enc_layers"] = complexity_values.n_enc_layers

        self._hparams["multiplier"] = self._trial.suggest_float(
            "multiplier", low=1e-2, high=10.0
        )
        self._hparams["dropout"] = self._trial.suggest_float(
            "dropout", low=0.0, high=0.5
        )

    def _register_optim_hparams(self):
        self._hparams["lr"] = self._trial.suggest_float(
            "lr", low=1e-5, high=1e-2, log=True
        )
        self._hparams["weight_decay"] = self._trial.suggest_float(
            "weight_decay", low=1e-5, high=1e-1, log=True
        )
        # betas?
        # warmup? -> TODO: need a flag to post-norm

    def _register_data_hparams(self):
        self._hparams["batch_size"] = self._trial.suggest_categorical(
            "batch_size", choices=[16, 32, 64]
        )
        # TODO: edge strategy?
        self._hparams["chunk_size"] = self._trial.suggest_int(
            "chunk_size", low=15, high=50, step=5
        )

    def _register_trainer_hparams(self):
        # these are for the lightning.Trainer
        # epochs maybe constant?
        # self._hparams["max_epochs"] = self._trial.suggest_int(
        #     "max_epochs", low=10, high=500, step=25
        # )
        # gradient clipping?
        pass

    def _register_loss_hparams(self):
        self._hparams["margin"] = self._trial.suggest_float(
            "margin", low=0.05, high=10.0
        )

    def _register_experiment_hparams(self):
        # self._hparams["patience"]
        if self._hparams["swa"]:
            self._hparams["swa_epoch_start"] = self._trial.suggest_float(
                "swa_epoch_start", low=0.5, high=0.8
            )
            self._hparams["annealing_epochs"] = self._trial.suggest_int(
                "annealing_epochs", low=10, high=100
            )
            self._hparams["annealing_strategy"] = self._trial.suggest_categorical(
                "annealing_strategy",
                choices=["linear", "cosine"],  # TODO: could get from typing.Literal
            )

    def _register_augmentation_hparams(self):
        self._hparams["sample_scale"] = self._trial.suggest_float(
            "sample_scale", low=1.0, high=15.0
        )
        self._hparams["sample_rate"] = self._trial.suggest_float(
            "sample_rate", low=0.25, high=0.75
        )


# x = Hyperparameter.from_value(value=1.2, value_range=(0.0, 5.0), tunable=True)
# y = Hyperparameter.from_value(value="A", categories=["A", "B", "C"])
