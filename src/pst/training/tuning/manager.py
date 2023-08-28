from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import optuna
from optuna.storages import RDBStorage
from optuna.trial import FrozenTrial, TrialState
from sqlite_utils import Database as _Database

FilePath = str | Path


@contextmanager
def Database(file: FilePath):
    database = _Database(file)
    try:
        yield database
    finally:
        database.close()


class StudyManager:
    ENGINE_KWARGS: dict[str, Any] = dict(connect_args=dict(check_same_thread=False))
    STUDY_NAME = "tuning"

    def __init__(self, storage_file: FilePath, **study_kwargs):
        self.storage_file = Path(storage_file)
        self.url = StudyManager.create_url(self.storage_file)

        study_kwargs["load_if_exists"] = True
        study_kwargs["storage"] = self.storage
        study_kwargs["study_name"] = StudyManager.STUDY_NAME
        self._study = optuna.create_study(**study_kwargs)
        self._seen = set()
        self._store_history()

    @staticmethod
    def _get_unique_fields(trial: FrozenTrial) -> tuple:
        return (
            trial.datetime_start,
            trial.datetime_complete,
            trial.value,
            tuple(trial.params.items()),
        )

    def _store_history(self):
        for trial in self._study.get_trials(deepcopy=False):
            uniq_fields = StudyManager._get_unique_fields(trial)
            self._seen.add(uniq_fields)

    @staticmethod
    def _get_study_name(db_file: FilePath) -> str:
        with Database(db_file) as db:
            study_names: list[str] = [row["study_name"] for row in db["studies"].rows]

        if len(study_names) > 1:
            raise ValueError(f"Multiple studies found in {db_file}")
        elif len(study_names) == 0:
            raise ValueError(f"No studies found in {db_file}")
        return study_names[0]

    def sync_study(
        self,
        other_study: optuna.Study,
        prune_failed_trials: bool = True,
        verbose: bool = False,
    ):
        n_trials = 0
        for trial in other_study.trials:
            if prune_failed_trials and trial.state == TrialState.FAIL:
                continue

            uniq_fields = StudyManager._get_unique_fields(trial)
            if uniq_fields not in self._seen:
                n_trials += 1
                self._study.add_trial(trial)
            self._seen.add(uniq_fields)

        if verbose:
            print(f"Added {n_trials} trials from {other_study.study_name}")

    def sync_file(self, other: FilePath, cleanup: bool = False, **kwargs):
        study_name = StudyManager._get_study_name(other)
        other_study = optuna.load_study(
            study_name=study_name,
            storage=StudyManager.create_url(other),
        )

        self.sync_study(other_study, **kwargs)

        if cleanup:
            other = Path(other)
            other.unlink()

    def sync_files(self, files: Iterable[FilePath], **kwargs):
        for file in files:
            self.sync_file(file, **kwargs)

    def sync_studies(self, studies: Iterable[optuna.Study], **kwargs):
        for study in studies:
            self.sync_study(study, **kwargs)

    @staticmethod
    def create_url(file: FilePath) -> str:
        return f"sqlite:///{file}"

    @property
    def storage(self) -> RDBStorage:
        return RDBStorage(url=self.url, engine_kwargs=StudyManager.ENGINE_KWARGS)

    @property
    def study_name(self) -> str:
        return self._study.study_name

    @property
    def study(self) -> optuna.Study:
        return self._study
