from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import Optional, cast

from optuna.trial import TrialState
from pydantic import DirectoryPath, Field
from pydantic_argparse import ArgumentParser, BaseCommand
from sqlite_utils import Database as _Database
from sqlite_utils.db import Table

from pst.training.tuning import TrialManager

FilePath = str | Path
TABLENAME = "trials"
TRIALTABLES = [
    "trial_values",
    "trial_params",
    "trial_intermediate_values",
    "trial_heartbeats",
    "trial_system_attributes",
    "trial_user_attributes",
]


class Args(BaseCommand):
    tuning_dir: DirectoryPath = Field(
        description="Existing directory of optuna history databases"
    )
    extension: str = Field("db", description="database file extension to glob")
    merged_file: Path = Field(
        Path("merged.db"), description="name of file after merging databases"
    )


def parse_args() -> Args:
    parser = ArgumentParser(model=Args)
    return parser.parse_typed_args()


def standardize_ext(ext: str) -> str:
    return f"*.{ext.lstrip('.')}"


@contextmanager
def Database(file: FilePath):
    database = _Database(file)
    try:
        yield database
    finally:
        database.close()


@contextmanager
def foreign_keys(db: _Database):
    try:
        db.execute("PRAGMA foreign_keys = ON")
        yield
    finally:
        db.execute("PRAGMA foreign_keys = OFF")


def state_name(state: TrialState) -> str:
    return repr(state).rsplit(".", 1)[-1]


def is_empty(db: _Database) -> bool:
    table = cast(Table, db[TABLENAME])
    # True (is_empty) is table doesn't exist or table exists but has no rows
    return not table.exists() or not bool(table.count)


def _cascade_delete_schema(table: Table) -> str:
    schema = table.schema

    new_schema: list[str] = list()
    for line in schema.splitlines():
        if line.lstrip().startswith("FOREIGN KEY"):
            line = f"{line} ON DELETE CASCADE"
        new_schema.append(line)
    create_table = "\n".join(new_schema)
    return create_table


def _update_schema_cascade_delete(db: _Database, table: Table):
    tmp = "_tmp"
    create_table = _cascade_delete_schema(table)
    db.rename_table(table.name, tmp)
    db.execute(create_table)
    db.execute(f"INSERT INTO {table.name} SELECT * FROM {tmp}")
    db[tmp].drop()
    # for some reason commit is required with sqlite-utils
    conn = cast(Connection, db.conn)
    conn.commit()


def check_cascade_delete(db: _Database):
    # TODO: this will copy data if the schema is wrong
    # so in future may be better to create an empty table with the correct schema
    # and then union the two tables
    for name in TRIALTABLES:
        table = cast(Table, db[name])
        schema = table.schema
        if "ON DELETE CASCADE" not in schema:
            _update_schema_cascade_delete(db, table)


def prune_failed_trials(table: Table):
    params = {"state": state_name(TrialState.FAIL)}
    table.delete_where("state = :state", params)


def cleanup(file: Path):
    with Database(file) as db:
        if is_empty(db):
            file.unlink()
            return

        # fix schemas for trial tables to cascade delete
        check_cascade_delete(db)

        # activate foreign keys in sqlite for cascading deletes
        with foreign_keys(db):
            table = cast(Table, db[TABLENAME])
            prune_failed_trials(table)
            conn = cast(Connection, db.conn)
            conn.commit()
            if is_empty(db):
                file.unlink()


def merge(merged_file: Path, files: list[Path]):
    TrialManager.with_files(local=merged_file, files=files).sync_files()


def main(args: Optional[Args] = None):
    if args is None:
        args = parse_args()

    ext = standardize_ext(args.extension)

    # clean up empty files or files with failed trials
    for file in args.tuning_dir.glob(ext):
        cleanup(file)

    # merge remaining files into a single db
    remaining_files = list(args.tuning_dir.glob(ext))
    merge(args.merged_file, remaining_files)

    # then remove the remaining files and keep only the merged file
    for file in remaining_files:
        file.unlink()


if __name__ == "__main__":
    main()
