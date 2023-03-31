from __future__ import annotations

from pathlib import Path
from shutil import copyfileobj, rmtree
from typing import Literal, Optional

import numpy as np
import lightning as L
import tables as tb
import torch
from numpy.typing import NDArray
from lightning.pytorch.callbacks import BasePredictionWriter

from pst.arch.lightning.data import GenomeSetDataModule
from pst.arch.lightning.modules import _ProteinSetTransformer


def sort_key(file: Path) -> int:
    """Return the trailing number on a file path if it exists

    Args:
        file (Path): any file path
    """
    return int(file.name.split(".")[0].rsplit("_", 1)[1])


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        outdir: Path,
        dataset: GenomeSetDataModule,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.outdir = outdir
        self.dataset = dataset
        self.compression = tb.Filters(complevel=4, complib="blosc:lz4")
        self.dataset_prefix = "dataset"
        self.batch_prefix = "batch"

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: _ProteinSetTransformer,
        predictions: torch.Tensor,
        batch_indices: Optional[list[int]],
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ):
        outdir = self.outdir.joinpath(f"{self.dataset_prefix}_{dataloader_idx}")
        outdir.mkdir(parents=True, exist_ok=True)
        output_file = outdir.joinpath(f"{self.batch_prefix}_{batch_idx}.h5")
        name_output = outdir.joinpath(f"{self.batch_prefix}_{batch_idx}.names.txt")
        predictions = predictions.numpy()

        batch_size = int(batch.size(0))
        names = self.dataset._dataset.convert_batch_idx_to_genome_names(
            batch_idx, batch_size
        )
        with tb.File(output_file, "w") as fp:
            fp.create_carray(
                "/",
                "data",
                obj=predictions,
                shape=predictions.shape,
                filters=self.compression,
            )

        with name_output.open("w") as fp:
            for name in names:
                fp.write(f"{name}\n")

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Combine all separate batch files into a single file

        Args:
            trainer (pl.Trainer): pytorch-lightning trainer
            pl_module (pl.LightningModule): pytorch-lightning module
        """
        super().on_predict_end(trainer, pl_module)

        dataset_paths = sorted(
            self.outdir.glob(f"*{self.dataset_prefix}*/"), key=sort_key
        )
        data_paths = {
            dataset_path: sorted(
                dataset_path.glob(f"*{self.batch_prefix}*.h5"), key=sort_key
            )
            for dataset_path in dataset_paths
        }
        name_paths = {
            dataset_path: sorted(
                dataset_path.glob(f"*{self.batch_prefix}*.txt"), key=sort_key
            )
            for dataset_path in dataset_paths
        }

        # copy .h5 embeddings arrays
        for dataset_path, batch_paths in data_paths.items():
            data_output = self.outdir.joinpath(f"{dataset_path.name}.h5")
            data: list[NDArray[np.float32]] = list()
            with tb.File(data_output, "w") as fdst:
                for data_path in batch_paths:
                    with tb.File(data_path) as fsrc:
                        data.append(fsrc.root.data[:])

                concat = np.vstack(data)
                fdst.create_carray(
                    "/",
                    "data",
                    obj=concat,
                    shape=concat.shape,
                    filters=self.compression,
                )

        # copy sequence names
        for dataset_path, batch_paths in name_paths.items():
            data_output = self.outdir.joinpath(f"{dataset_path.name}.names.txt")
            with data_output.open("wb") as fdst:
                for name_path in batch_paths:
                    with name_path.open("rb") as fsrc:
                        copyfileobj(fsrc, fdst)

        # delete dataset files
        for dataset_path in dataset_paths:
            rmtree(dataset_path)
