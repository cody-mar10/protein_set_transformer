from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
from typing import Any, Optional, Sequence

from optuna.storages import RDBStorage
from optuna.storages._rdb.models import (
    TrialIntermediateValueModel,
    TrialModel,
    TrialParamModel,
    TrialValueModel,
)
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.engine import create_engine as _create_engine
from sqlalchemy.orm import scoped_session as _scoped_session
from sqlalchemy.orm import sessionmaker
from sqlite_utils import Database as _Database

FilePath = str | Path

_TABLES = [TrialModel, TrialParamModel, TrialValueModel, TrialIntermediateValueModel]
_TABLENAMES: set[str] = {table.__tablename__ for table in _TABLES}


class TrialManager:
    current_values: dict[str, int]
    primary_keys: dict[Any, str]
    trial_id_remap: dict[int, int]
    TABLES = _TABLES
    TABLENAMES = _TABLENAMES
    TRIAL_ID = "trial_id"
    TRIAL_NUMBER = "number"
    ENGINE_KWARGS: dict[str, Any] = dict(connect_args=dict(check_same_thread=False))
    OTHER = "other"

    def __init__(
        self, master_file: FilePath, _files: Optional[Sequence[FilePath]] = None
    ):
        self.master_file = Path(master_file)
        self.master_url = TrialManager.create_url(self.master_file)
        self.master_engine = TrialManager.create_engine(url=self.master_url)
        if self.master_file.exists():
            self._set_initial_values()
        else:
            self._reset_values()

        self._files = _files or []

    @classmethod
    def with_files(cls, local: FilePath, files: Sequence[FilePath]):
        other_files = None
        local = Path(local)
        if files and not local.exists():
            # just choose the first file as the master
            # it will just not contribute anything later
            master_file, *other_files = files

            # copy chosen master_file to local
            copyfile(master_file, local)
        return cls(local, _files=other_files)

    def _reset_values(self):
        self.current_values = dict()
        self.primary_keys = dict()
        self.trial_id_remap = dict()

        for table in TrialManager.TABLES:
            for col in table.__table__.columns:
                if col.primary_key:
                    primary_key = col.name
                    self.primary_keys[table] = primary_key
                    break
            else:
                raise RuntimeError(f"Table {table.__tablename__} has no primary key")

            self.current_values[primary_key] = 1

    def _set_initial_values(self):
        self._reset_values()
        with self.master_session() as session:
            for table in TrialManager.TABLES:
                primary_key = self.primary_keys[table]

                last_row = session.scalars(
                    select(table).order_by(-getattr(table, primary_key))
                ).first()
                # defaults to -1 so current_value points to 0
                last_value = (
                    int(getattr(last_row, primary_key)) if last_row is not None else -1
                )
                self.current_values[primary_key] = last_value + 1

    @staticmethod
    def _find_table_differences(tablename: str) -> str:
        if tablename not in TrialManager.TABLENAMES:
            raise ValueError(f"{tablename=} must be in {TrialManager.TABLENAMES}")

        query = f"""
        SELECT * FROM main.{tablename} 
        EXCEPT 
        SELECT * FROM {TrialManager.OTHER}.{tablename} 
        UNION ALL 
        SELECT * FROM {TrialManager.OTHER}.{tablename} 
        EXCEPT 
        SELECT * FROM main.{tablename} 
        """
        return query

    def _sync(self, other: FilePath):
        with self.master_database() as db:
            db.attach(alias=TrialManager.OTHER, filepath=other)
            for table in TrialManager.TABLES:
                new_records: list[dict[str, Any]] = list()

                tablename = table.__tablename__
                primary_key = self.primary_keys[table]
                query = self._find_table_differences(tablename)
                diff_records = db.query(query)

                for record in diff_records:
                    try:
                        trial_id = record[TrialManager.TRIAL_ID]
                    except KeyError as e:
                        raise RuntimeError(
                            f"Invalid optuna trial history database: {other}"
                        ) from e

                    record[TrialManager.TRIAL_ID] = self.new_trial(trial_id)

                    if TrialManager.TRIAL_NUMBER in record:
                        record[TrialManager.TRIAL_NUMBER] = (
                            record[TrialManager.TRIAL_ID] - 1
                        )

                    if primary_key != TrialManager.TRIAL_ID:
                        record[primary_key] = self.current_value_and_increment(
                            primary_key
                        )

                    new_records.append(record)
                db[tablename].insert_all(new_records)  # type: ignore

    def sync_files(self):
        for file in self._files:
            self._sync(file)
            # this is on a per db basis so needs to be cleared for each file
            self.trial_id_remap.clear()

    @property
    def current_trial_id(self) -> int:
        return self.current_values[TrialManager.TRIAL_ID]

    def increment_current_trial_id(self):
        self.current_values[TrialManager.TRIAL_ID] += 1

    def new_trial(self, trial_id: int) -> int:
        if trial_id not in self.trial_id_remap:
            self.trial_id_remap[trial_id] = self.current_trial_id
            self.increment_current_trial_id()
        return self.trial_id_remap[trial_id]

    def current_value(self, primary_key: str) -> int:
        return self.current_values[primary_key]

    def current_value_and_increment(self, primary_key: str) -> int:
        value = self.current_value(primary_key)
        self.current_values[primary_key] += 1
        return value

    @property
    def url(self) -> str:
        return self.master_url

    @staticmethod
    def create_url(file: FilePath) -> str:
        return f"sqlite:///{file}"

    @staticmethod
    def create_engine(
        file: Optional[FilePath] = None, url: Optional[str] = None
    ) -> Engine:
        if url is None:
            if file is not None:
                url = TrialManager.create_url(file)
            else:
                raise ValueError(
                    "Either the `optuna` database filepath or a SQL URL must be passed"
                )
        return _create_engine(url, **TrialManager.ENGINE_KWARGS)

    @staticmethod
    @contextmanager
    def scoped_session(engine: Engine):
        ScopedSession = _scoped_session(
            sessionmaker(bind=engine, autoflush=True, autocommit=False)
        )
        try:
            yield ScopedSession()
        finally:
            ScopedSession.remove()

    def master_session(self):
        return TrialManager.scoped_session(self.master_engine)

    @staticmethod
    @contextmanager
    def Database(file: FilePath):
        database = _Database(file)
        try:
            yield database
        finally:
            database.close()

    def master_database(self):
        return TrialManager.Database(file=self.master_file)

    @property
    def storage(self) -> RDBStorage:
        return RDBStorage(url=self.url, engine_kwargs=TrialManager.ENGINE_KWARGS)
