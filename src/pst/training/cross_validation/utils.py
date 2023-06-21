from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class _CVRecord:
    fold: int
    metric: str
    step: int
    value: float


_SummaryType = list[_CVRecord]


class CrossValEventSummarizer:
    def __init__(self, root_dir: Path, name: str) -> None:
        self.logdir = root_dir.joinpath(name)

    @staticmethod
    def load_events(version: Path) -> EventAccumulator:
        events = EventAccumulator(path=version.as_posix(), size_guidance=0)
        events.Reload()
        return events

    def summarize(
        self, metric: str = "val_loss", step: Literal["epoch", "step"] = "epoch"
    ) -> pd.DataFrame:
        metric_name = f"{metric}_{step}"

        summary: _SummaryType = list()
        for version in self.logdir.glob("version_*"):
            fold_idx = int(version.stem.rsplit("_", maxsplit=1)[-1])
            events = self.load_events(version)
            for event in events.Scalars(metric_name):
                record = _CVRecord(
                    fold=fold_idx,
                    metric=metric_name,
                    step=event.step,
                    value=float(event.value),
                )
                summary.append(record)

        df = (
            pd.DataFrame(summary)
            .sort_values(by=["fold", "step"])
            .reset_index(drop=True)
        )
        # need to normalize the step no. since all folds will have a diff number of steps
        minmax = df.groupby("fold")["step"].agg(["min", "max"]).reset_index()
        df = (
            df.merge(minmax)
            .assign(
                diff=lambda df: df["max"] - df["min"],
                norm_step=lambda df: (df["step"] - df["min"]) / df["diff"],
            )
            .drop(columns=["diff"])
            .rename(columns={"min": "min_step", "max": "max_step"})
        )
        return df

    def save(self, summary: pd.DataFrame, output_name: str):
        output = self.logdir.joinpath(output_name)
        summary.to_csv(output, sep="\t", index=False)

    def summarize_and_save(
        self,
        output_name: str,
        metric: str = "val_loss",
        step: Literal["epoch", "step"] = "epoch",
    ):
        summary = self.summarize(metric=metric, step=step)
        self.save(summary=summary, output_name=output_name)
