#!/usr/bin/env python3

import shlex
import subprocess
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ByteSize, Field, ValidationError, field_validator
from pydantic_argparse import ArgumentParser

DEVICES = 1
DurationOpts = Literal["short", "medium", "long"]


class Args(BaseModel):
    input: Path = Field(..., description="input data .h5 file")
    outdir_base: str = Field(
        ...,
        description="base name of the output directory, will be appended with an exp number",  # noqa: E501
    )
    output: Path = Field(
        Path("tuning_inputs.csv"), description="tuning inputs file for job submission"
    )
    epochs: int = Field(
        75, description="number of epochs to run per unit training session", gt=0
    )
    memory: str = Field("35GB", description="memory requested")
    disk: str = Field("20GB", description="disk space requested")
    n_trials: int = Field(
        3, description="number of tuning trials to run for a single job instance", gt=0
    )
    jobs: int = Field(25, description="number of tuning jobs to run", gt=0)
    submit: Optional[Path] = Field(
        None,
        description="submission condor file to be passed to condor_submit if supplied",
    )
    duration: DurationOpts = Field(
        "long", description="CHTC job length: [short=12h, medium=24h, long=7d]"
    )

    @field_validator("memory", "disk", mode="before")
    def validate_byte_size_argument(cls, value: str) -> str:
        try:
            ByteSize._validate(
                value,
                "",  # type: ignore - arg not used
            )
        except ValidationError as err:
            print(err)
            raise err
        else:
            return value


def parse_args() -> Args:
    parser = ArgumentParser(model=Args)
    return parser.parse_typed_args()


def submit(submit_file: Path):
    command = f"condor_submit {submit_file}"
    subprocess.run(shlex.split(command))


def main():
    args = parse_args()

    with args.output.open("w") as fp:
        for i in range(args.jobs):
            line = [
                f"{args.input}",
                f"{args.outdir_base}_{i}",
                f"{DEVICES}",
                f"{args.epochs}",
                args.memory,
                args.disk,
                args.duration,
                f"{args.n_trials}",
                f"exp{i}",
            ]
            csvline = ",".join(line)
            fp.write(f"{csvline}\n")

    if args.submit is not None:
        submit(args.submit)


if __name__ == "__main__":
    main()
