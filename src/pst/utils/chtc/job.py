#!/usr/bin/env python3

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEVICES = 1
DURATION = "long"


@dataclass
class Args:
    input: Path
    outdir_base: str
    output: Path
    epochs: int
    memory: str
    disk: str
    n_trials: int
    jobs: int
    submit: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        type=Path,
        required=True,
        help="input data .h5 file",
    )
    parser.add_argument(
        "-o",
        "--outdir-base",
        required=True,
        help="base name of the output directory, will be appended with an exp number",
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="FILE",
        type=Path,
        default=Path("tuning_inputs.csv"),
        help="tuning inputs file for job submission (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="INT",
        type=int,
        default=75,
        help="number of epochs to run per unit training session (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--memory",
        metavar="BYTES",
        default="35GB",
        help="memory requested (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--disk",
        metavar="BYTES",
        default="20GB",
        help="disk space requested (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--n-trials",
        metavar="INT",
        type=int,
        default=3,
        help=(
            "number of tuning trials to run for a single job instance "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        metavar="INT",
        type=int,
        default=25,
        help="number of tuning jobs to run (default: %(default)s)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit condor job (ie condor_submit)",
    )

    parsed_args = parser.parse_args()
    args = Args(
        input=parsed_args.input,
        outdir_base=parsed_args.outdir_base,
        output=parsed_args.file,
        epochs=parsed_args.epochs,
        memory=parsed_args.memory,
        disk=parsed_args.disk,
        n_trials=parsed_args.n_trials,
        jobs=parsed_args.jobs,
        submit=parsed_args.submit,
    )

    return args


def submit(output: Path):
    command = f"condor_submit {output}"
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
                f"{args.memory}",
                f"{args.disk}",
                DURATION,
                f"{args.n_trials}",
                f"exp{i}",
            ]
            csvline = ",".join(line)
            fp.write(f"{csvline}\n")

    if args.submit:
        submit(args.output)


if __name__ == "__main__":
    main()
