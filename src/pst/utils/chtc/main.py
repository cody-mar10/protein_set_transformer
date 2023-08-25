from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_argparse import ArgumentParser

from pst.utils.chtc import cleanup, job
from pst.utils.chtc.cleanup import Args as CleanupArgs
from pst.utils.chtc.job import Args as JobArgs


class Args(BaseModel):
    job: Optional[JobArgs] = Field(
        None, description="create and submit tuning jobs in CHTC"
    )
    cleanup: Optional[CleanupArgs] = Field(
        None, description="merge and cleanup tuning databases"
    )


def parse_args() -> Args:
    parser = ArgumentParser(
        model=Args,
        description=(
            "CHTC utilities: (1) Create and submit tuning jobs, "
            "and (2) merge and cleanup tuning databases"
        ),
    )

    return parser.parse_typed_args()


def main():
    args = parse_args()

    if args.job:
        job.main(args.job)
    elif args.cleanup:
        cleanup.main(args.cleanup)


if __name__ == "__main__":
    main()
