import importlib.metadata
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

PST_VERSION = importlib.metadata.version("ptn-set-transformer")


class Version(tb.IsDescription):
    version = tb.StringCol(itemsize=20)  # type: ignore


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

        self._write_version_table()

        # create an empty array on disk
        # then just add the embeddings to the appropriate place as they are computed
        # (which happens potentially out of order)
        self.output = self._file.create_carray(
            self._file.root,
            "data",
            atom=tb.Float32Atom(),
            shape=(dataset.num_sequences, model.model.embed_dim),
            filters=self.compression,
        )

        self.seq2order = dataset.seq2order

    def _write_version_table(self):
        table = self._file.create_table(
            where=self._file.root,
            name="_version",
            description=Version,  # type: ignore
        )

        table.row["version"] = PST_VERSION
        table.row.append()
        table.flush()

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
        seqnames, seqs, tokens = batch

        sequence_order: NDArray[np.int64] = np.array(
            [self.seq2order[name] for name in seqnames]
        )

        data: NDArray[np.float32] = predictions.detach().cpu().numpy()
        self.output[sequence_order, :] = data  # type: ignore

    def on_predict_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        pl_module = cast(ESM2, pl_module)
        self._file.close()
        model_name = pl_module.LAYERS_TO_MODELNAME[pl_module.repr_layers]
        output_name = f"{model_name}_results.h5"
        self._output.rename(self.outdir.joinpath(output_name))
