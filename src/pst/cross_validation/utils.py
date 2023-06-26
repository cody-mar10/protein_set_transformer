from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    ScalarEvent,
)


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
        step2epoch: list[tuple[int, int, int]] = list()
        for version in self.logdir.glob("version_*"):
            fold_idx = int(version.stem.rsplit("_", maxsplit=1)[-1])
            events = self.load_events(version)
            event: ScalarEvent
            for event in events.Scalars(metric_name):
                record = _CVRecord(
                    fold=fold_idx,
                    metric=metric_name,
                    step=event.step,
                    value=float(event.value),
                )
                summary.append(record)

            for event in events.Scalars("epoch"):
                record = (fold_idx, event.step, int(event.value))
                step2epoch.append(record)

        # for some reason the same step may be logged multiple times
        epochs: pd.DataFrame = pd.DataFrame(
            step2epoch, columns=["fold", "step", "epoch"]
        ).drop_duplicates()

        # epochs should be a constant roughly across all folds
        df = (
            pd.DataFrame(summary)
            .merge(epochs, on=["fold", "step"])
            .sort_values(by=["fold", "step"])
            .reset_index(drop=True)
        )

        # TODO: report avg val loss at final epoch?

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


class CVStatusLogger(L.Callback):
    _HEADER = f"{'-' * 100}\nCV TRAINING INFO:\n"
    _FOOTER = f"{'*' * 100}\n"

    def __init__(self, fold_idx: int, train_group_ids: list[int], val_group_id: int):
        super().__init__()
        self.fold_idx = fold_idx
        self.train_group_ids = train_group_ids
        self.val_group_id = val_group_id

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        super().on_fit_start(trainer, pl_module)
        msg = (
            f"{self._HEADER}Fold: {self.fold_idx} "
            f"Training groups: {self.train_group_ids} "
            f"Val group: {self.val_group_id}"
        )
        print(msg)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        super().on_fit_end(trainer, pl_module)
        msg = f"Fold: {self.fold_idx} TRAINING COMPLETE\n{self._FOOTER}"
        print(msg)
