from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Sequence

import lightning as L
import tables as tb
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from pst.data.modules import GenomeDataModule
from pst.nn.modules import ProteinSetTransformer
from pst.typing import NumpyFloat32, PairTensor

logger = logging.getLogger(__name__)


def sort_key(file: Path) -> int:
    """Return the trailing number on a file path if it exists

    Args:
        file (Path): any file path
    """
    return int(file.name.split(".")[0].rsplit("_", 1)[1])


class PredictionWriter(BasePredictionWriter):
    OUTPUTS = ["data", "attn"]

    def __init__(
        self,
        outdir: Path,
        model: ProteinSetTransformer,
        datamodule: GenomeDataModule,
        output_name: str = "predictions.h5",
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.outdir = outdir
        self.datamodule = datamodule

        self.compression = tb.Filters(complevel=4, complib="blosc:lz4")
        self._file = tb.File(self.outdir.joinpath(output_name), "w")
        self._arrays: dict[str, tb.EArray] = dict()

        n_nodes = self.datamodule.dataset.data.shape[0]
        n_graphs = int(self.datamodule.dataset.sizes.numel())
        for output in PredictionWriter.OUTPUTS:
            if output == "data":
                n = n_graphs
                shape = (0, model.config.out_dim)
            else:
                n = n_nodes
                shape = (0, model.config.num_heads)

            array = self._file.create_earray(
                self._file.root,
                output,
                atom=tb.Float32Atom(),
                shape=shape,
                expectedrows=n,
                filters=self.compression,
            )

            self._arrays[output] = array

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: PairTensor,
        batch_indices: Optional[Sequence[int]],
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # results should come in the same order as the data is stored in the dataset
        data, attn = predictions
        np_data: NumpyFloat32 = data.cpu().numpy()
        np_attn: NumpyFloat32 = attn.cpu().numpy()

        for name, array in zip(PredictionWriter.OUTPUTS, [np_data, np_attn]):
            self._arrays[name].append(array)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        num_graphs = self._arrays["data"].shape[0]  # type: ignore
        if num_graphs != int(self.datamodule.dataset.sizes.numel()):
            raise RuntimeError(
                f"Expected {self.datamodule.dataset.sizes.numel()} graphs in output but "
                f"got {num_graphs}"
            )

        num_nodes = self._arrays["attn"].shape[0]  # type: ignore
        if num_nodes != int(self.datamodule.dataset.data.shape[0]):
            raise RuntimeError(
                f"Expected {self.datamodule.dataset.data.shape[0]} nodes in output but "
                f"got {num_nodes}"
            )

        self._file.close()

        logger.info(f"Saved predictions to {self.outdir}")
