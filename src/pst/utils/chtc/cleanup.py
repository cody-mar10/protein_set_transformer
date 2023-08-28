from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from pydantic import DirectoryPath, Field
from pydantic_argparse import ArgumentParser, BaseCommand

from pst.training.tuning import StudyManager

FilePath = str | Path


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


def merge(merged_file: Path, files: Iterable[Path]):
    study_manager = StudyManager(merged_file)
    study_manager.sync_files(
        files, prune_failed_trials=True, verbose=True, cleanup=True
    )


def main(args: Optional[Args] = None):
    if args is None:
        args = parse_args()

    ext = standardize_ext(args.extension)

    # merge remaining files into a single db AND remove the other files
    other_files = (
        file for file in args.tuning_dir.glob(ext) if file != args.merged_file
    )

    merge(args.merged_file, other_files)


if __name__ == "__main__":
    main()
