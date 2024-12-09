import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

DEVICES = 1


class DurationOpts(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class Main:
    @staticmethod
    def _check_str_is_bytesize(value: str):
        if value[-1] != "B":
            raise ValueError(f"byte size must end with 'B': {value}")

        second_to_last = value[-2]
        if not second_to_last.isdigit() and second_to_last not in "KMGT":
            raise ValueError(
                f"The allow order of magnitude prefixes are any of 'KMGT', received: {value}"
            )

    @staticmethod
    def _validate_positive_ints(**kwargs):
        for name, arg in kwargs.items():
            if arg < 1:
                raise ValueError(f"{name} must be >= 1, received: {arg}")

    @staticmethod
    def _submit(submit_file: Path):
        command = f"condor_submit {submit_file}"
        subprocess.run(shlex.split(command))

    def job(
        self,
        input: Path,
        outdir_base: str,
        output: Path = Path("tuning_inputs.csv"),
        epochs: int = 75,
        memory: str = "35GB",
        disk: str = "20GB",
        n_trials: int = 3,
        jobs: int = 25,
        submit: Optional[Path] = None,
        duration: DurationOpts = DurationOpts.long,
    ):
        """Create and submit tuning jobs in CHTC

        Args:
            input (Path): input data .h5 file
            outdir_base (str): base name of the output directory, will be appended with an exp number
            output (Path): tuning inputs file for job submission.
            epochs (int): number of epochs to run per unit training session.
            memory (str): memory requested.
            disk (str): disk space requested.
            n_trials (int): number of tuning trials to run for a single job instance
            jobs (int): number of tuning jobs to run. Defaults to 25.
            submit (Optional[Path]): submission condor file to be passed to condor_submit if supplied.
            duration (DurationOpts): CHTC job length: [short=12h, medium=24h, long=7d].
        """
        #### validate inputs
        self._validate_positive_ints(epochs=epochs, n_trials=n_trials, jobs=jobs)
        self._check_str_is_bytesize(memory)
        self._check_str_is_bytesize(disk)

        with output.open("w") as fp:
            for i in range(jobs):
                line = [
                    f"{input}",
                    f"{outdir_base}_{i}",
                    f"{DEVICES}",
                    f"{epochs}",
                    memory,
                    disk,
                    duration.value,
                    f"{n_trials}",
                    f"exp{i}",
                ]
                csvline = ",".join(line)
                fp.write(f"{csvline}\n")

        if submit is not None:
            self._submit(submit)
