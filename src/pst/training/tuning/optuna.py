from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import optuna

from pst.utils.cli.experiment import ExperimentArgs

from .manager import StudyManager

OptunaCallback = Callable[[optuna.Study, optuna.trial.FrozenTrial], None]
ObjectiveFuncType = Callable[[optuna.Trial], float | Sequence[float]]


class OptunaIntegration:
    STUDY_KWARGS = {
        "direction": "minimize",
        "study_name": StudyManager.STUDY_NAME,
        "load_if_exists": True,
    }

    def __init__(self, expt_cfg: ExperimentArgs, default_root_dir: Path):
        self._config = expt_cfg
        self.pruner = (
            optuna.pruners.MedianPruner(
                n_startup_trials=expt_cfg.pruning_warmup_trials,
                n_warmup_steps=expt_cfg.pruning_warmup_steps,
            )
            if expt_cfg.prune
            else optuna.pruners.NopPruner()
        )
        self.root = default_root_dir
        self._study_manager = self._get_study_manager()

        self._callbacks: list[OptunaCallback] = list()

    @property
    def local_storage(self):
        storage_fname = f"{self._config.name}.db"
        storage_dir = self.root / self._config.name
        if not storage_dir.exists():
            storage_dir.mkdir(exist_ok=True, parents=True)
        return storage_dir.joinpath(storage_fname)

    def _get_study_manager(self) -> StudyManager:
        study_kwargs = OptunaIntegration.STUDY_KWARGS | {"pruner": self.pruner}
        study_manager = StudyManager(self.local_storage, **study_kwargs)
        if self._config.tuning_dir is None:
            return study_manager

        tuning_dbs = self._config.tuning_dir.glob("*.db")
        study_manager.sync_files(
            files=tuning_dbs,
            prune_failed_trials=True,
            cleanup=False,
            verbose=True,
        )

        return study_manager

    @property
    def study_manager(self) -> StudyManager:
        return self._study_manager

    @property
    def study(self) -> optuna.Study:
        return self._study_manager.study

    def register_callbacks(self, callbacks: OptunaCallback | Iterable[OptunaCallback]):
        if not isinstance(callbacks, Iterable):
            callbacks = [callbacks]
        self._callbacks.extend(callbacks)

    def optimize(self, fn: ObjectiveFuncType):
        callbacks = self._callbacks or None

        self.study.optimize(
            func=fn,
            n_trials=self._config.n_trials,
            callbacks=callbacks,
            # only need to GC if we're running multiple trials
            gc_after_trial=self._config.n_trials > 1,
        )
