from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import optuna
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

FilePath = str | Path


class StudyManager:
    ENGINE_KWARGS: dict[str, Any] = dict(connect_args=dict(check_same_thread=False))
    STUDY_NAME = "tuning"

    def __init__(
        self,
        storage_file: FilePath,
        prune_failed_trials: bool = False,
        failed_trial_retries: int = 2,
        **study_kwargs,
    ):
        self.storage_file = Path(storage_file)
        self.url = StudyManager.create_url(self.storage_file)
        self.prune_failed_trials = prune_failed_trials

        # TODO: need to handle request for a failed trial
        # ie, need to get the model, trainer, and tuner states back
        self._storage = StudyManager.create_storage(
            self.url,
            failed_trial_callback=RetryFailedTrialCallback(
                max_retry=failed_trial_retries
            ),
        )
        study_kwargs["load_if_exists"] = True
        study_kwargs["storage"] = self.storage
        study_kwargs["study_name"] = StudyManager.STUDY_NAME
        self._study = optuna.create_study(**study_kwargs)
        self._seen = set()
        self._store_history()

    @staticmethod
    def _get_unique_fields(trial: FrozenTrial) -> tuple:
        # including the trial value and sampled params is problematic for
        # hashing/comparing floats -> so just focus on datetime specifically
        # highly unlikely two independent trials share exact same datetime
        return (trial.datetime_start, trial.datetime_complete)

    def _store_history(self):
        for trial in self._study.get_trials(deepcopy=False):
            uniq_fields = StudyManager._get_unique_fields(trial)
            self._seen.add(uniq_fields)

    @staticmethod
    def get_all_trials(db_file: FilePath) -> Iterator[FrozenTrial]:
        storage = StudyManager.create_storage(StudyManager.create_url(db_file))
        for study in storage.get_all_studies():
            # have to do this with storage again since study is a FrozenStudy
            # which doesn't have the get_trials method
            for trial in storage.get_all_trials(
                study_id=study._study_id, deepcopy=True
            ):
                yield trial

    def sync_trials(
        self,
        other_study_trials: Iterable[FrozenTrial],
        verbose: bool = False,
        identifier: Optional[str | FilePath | Study] = None,
    ):
        n_trials = 0
        for trial in other_study_trials:
            if self.prune_failed_trials and trial.state == TrialState.FAIL:
                continue

            uniq_fields = StudyManager._get_unique_fields(trial)
            if uniq_fields not in self._seen:
                n_trials += 1
                self._study.add_trial(trial)
            self._seen.add(uniq_fields)

        if verbose:
            msg = f"Added {n_trials} trial{'s' if n_trials > 1 else ''}"
            addon = ""
            if isinstance(identifier, Study):
                addon = f" from {identifier.study_name}"
            else:
                addon = f" from {identifier}"
            msg = f"{msg}{addon}"
            print(msg)

    def sync_file(self, other: FilePath, cleanup: bool = False, verbose: bool = False):
        trials = StudyManager.get_all_trials(other)
        self.sync_trials(trials, verbose=verbose, identifier=other)

        if cleanup:
            other = Path(other)
            other.unlink()

    def sync_files(
        self, files: Iterable[FilePath], cleanup: bool = False, verbose: bool = False
    ):
        for file in files:
            self.sync_file(file, cleanup=cleanup, verbose=verbose)

    def sync_study(self, study: Study, verbose: bool = False):
        self.sync_trials(study.trials, verbose=verbose, identifier=study)

    def sync_studies(self, studies: Iterable[Study], verbose: bool = False):
        for study in studies:
            self.sync_study(study, verbose=verbose)

    @staticmethod
    def create_url(file: FilePath) -> str:
        return f"sqlite:///{file}"

    @staticmethod
    def create_storage(
        url: str, engine_kwargs: Optional[dict[str, Any]] = None, **kwargs
    ) -> RDBStorage:
        if engine_kwargs is None:
            engine_kwargs = StudyManager.ENGINE_KWARGS
        else:
            engine_kwargs = StudyManager.ENGINE_KWARGS | engine_kwargs
        return RDBStorage(url=url, engine_kwargs=engine_kwargs, **kwargs)

    @property
    def storage(self) -> RDBStorage:
        return self._storage

    @property
    def study_name(self) -> str:
        return self._study.study_name

    @property
    def study(self) -> Study:
        return self._study
