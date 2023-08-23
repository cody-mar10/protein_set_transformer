from __future__ import annotations

from pathlib import Path

import optuna

from pst.utils.cli.experiment import ExperimentArgs

from .manager import TrialManager


class OptunaIntegration:
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

    def _local_storage(self):
        # TODO: optuna is working on a journaling fs that may solve this
        storage_fname = f"{self._config.name}.db"
        storage_dir = self.root / self._config.name
        storage_dir.mkdir(exist_ok=True, parents=True)
        return storage_dir.joinpath(storage_fname)

    def _trial_manager(self) -> TrialManager:
        local_storage = self._local_storage()
        if self._config.tuning_dir is None:
            return TrialManager(local_storage)

        tuning_dbs = list(self._config.tuning_dir.glob("*.db"))
        trial_manager = TrialManager.with_files(local=local_storage, files=tuning_dbs)
        trial_manager.sync_files()

        return trial_manager

    def study(self) -> optuna.Study:
        study = optuna.create_study(
            storage=self._trial_manager().storage,
            direction="minimize",
            study_name="tuning",  # TODO: may need to update programatically
            pruner=self.pruner,
            load_if_exists=True,
        )
        return study
