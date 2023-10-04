from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from typing import Literal, Optional

import lightning as L
import numpy as np
import tables as tb
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from pst.data.modules import GenomeDataModule
from pst.nn.modules import ProteinSetTransformer
from pst.typing import NumpyFloat32, PairTensor


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
        datamodule: GenomeDataModule,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.outdir = outdir
        self.datamodule = datamodule
        self.compression = tb.Filters(complevel=4, complib="blosc:lz4")
        self.dataset_prefix = "dataset"
        self.batch_prefix = "batch"

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: ProteinSetTransformer,
        predictions: PairTensor,
        batch_indices: Optional[list[int]],
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # results should come in order
        outdir = self.outdir.joinpath(f"{self.dataset_prefix}_{dataloader_idx}")
        outdir.mkdir(parents=True, exist_ok=True)
        output_file = outdir.joinpath(f"{self.batch_prefix}_{batch_idx}.h5")

        data, attn = predictions
        np_data: NumpyFloat32 = data.cpu().numpy()
        np_attn: NumpyFloat32 = attn.cpu().numpy()

        with tb.File(output_file, "w") as fp:
            fp.create_carray(
                "/",
                "data",
                obj=np_data,
                shape=np_data.shape,
                filters=self.compression,
            )

            fp.create_carray(
                "/",
                "attn",
                obj=np_attn,
                shape=np_attn.shape,
                filters=self.compression,
            )

    def _concat_data(self, fdst: tb.File, data: list[NumpyFloat32], name: str):
        concat = np.vstack(data)
        fdst.create_carray(
            "/",
            name,
            obj=concat,
            shape=concat.shape,
            filters=self.compression,
        )

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

        # copy .h5 embeddings arrays
        for dataset_path, batch_paths in data_paths.items():
            data_output = self.outdir.joinpath(f"{dataset_path.name}.h5")

            data: list[NumpyFloat32] = list()
            attn: list[NumpyFloat32] = list()
            with tb.File(data_output, "w") as fdst:
                for data_path in batch_paths:
                    with tb.File(data_path) as fsrc:
                        data.append(fsrc.root.data[:])
                        attn.append(fsrc.root.attn[:])

                self._concat_data(fdst=fdst, data=data, name="data")
                self._concat_data(fdst=fdst, data=attn, name="attn")

        # delete dataset files
        for dataset_path in dataset_paths:
            rmtree(dataset_path)
