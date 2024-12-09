import warnings
from pathlib import Path
from typing import Iterable

from optuna.exceptions import ExperimentalWarning

from pst.training.tuning.manager import StudyManager

# ignore optuna experimental warnings since StudyManager uses RetryFailedTrialCallback
# but doesn't matter just for cleanup purposes
warnings.filterwarnings("ignore", category=ExperimentalWarning)


def standardize_ext(ext: str) -> str:
    return f"*.{ext.lstrip('.')}"


def merge(merged_file: Path, files: Iterable[Path], prune_failed_trials: bool = False):
    study_manager = StudyManager(merged_file, prune_failed_trials=prune_failed_trials)
    study_manager.sync_files(files, verbose=True, cleanup=True)


class Main:
    def cleanup(
        self,
        tuning_dir: Path,
        extension: str = "db",
        merged_file: Path = Path("merged.db"),
        prune_failed_trials: bool = False,
    ):
        """Cleanup optuna history databases by merging them into a single database

        Args:
            tuning_dir (Path): Existing directory of optuna history databases.
            extension (str): database file extension to glob.
            merged_file (Path): name of file after merging databases.
            prune_failed_trials (bool): whether to prune failed trials. Note: stopped trials
                are stored as failed, so these trials can't be resumed.
        """
        ##### validate inputs
        if not tuning_dir.exists():
            raise FileNotFoundError(f"{tuning_dir} does not exist")

        ext = standardize_ext(extension)

        # merge remaining files into a single db AND remove the other files
        other_files = (file for file in tuning_dir.glob(ext) if file != merged_file)

        merge(merged_file, other_files, prune_failed_trials=prune_failed_trials)
