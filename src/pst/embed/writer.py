from pathlib import Path
from typing import Literal, Optional, Sequence, cast

import lightning as L
import numpy as np
import tables as tb
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from numpy.typing import NDArray

from pst.embed.data import SequenceDataset
from pst.embed.model import ESM2, BatchType


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        outdir: Path,
        model: ESM2,
        dataset: SequenceDataset,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.outdir = outdir
        self.compression = tb.Filters(complevel=4, complib="blosc:lz4")
        self._output = self.outdir.joinpath("predictions.h5")
        self._file = tb.File(self._output, "w")
        self.output = self._file.create_earray(
            self._file.root,
            "data",
            atom=tb.Float32Atom(),
            shape=(0, model.model.embed_dim),
            expectedrows=len(dataset),
            filters=self.compression,
        )

    # TODO: should probably write the sequence names...
    # but for now just require users to keep track of the source fasta file
    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: torch.Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: BatchType,
        batch_idx: int,
        dataloader_idx: int,
    ):
        data: NDArray[np.float32] = predictions.detach().cpu().numpy()
        self.output.append(data)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pl_module = cast(ESM2, pl_module)
        self._file.close()
        model_name = pl_module.LAYERS_TO_MODELNAME[pl_module.repr_layers]
        output_name = f"{model_name}_results.h5"
        self._output.rename(self.outdir.joinpath(output_name))
