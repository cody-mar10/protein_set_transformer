from __future__ import annotations

from typing import Any

import lightning as L

from pst.arch.data import GenomeDataModule
from pst.arch.modules import ProteinSetTransformer

from .writer import PredictionWriter


# TODO: no need for predict kwargs -> change to actual values,
class Predictor:
    def __init__(
        self,
        predict: dict[str, Any],
        data: dict[str, Any],
        trainer: dict[str, Any],
    ):
        self.model = ProteinSetTransformer.load_from_checkpoint(predict["checkpoint"])
        self.datamodule = GenomeDataModule(shuffle=False, **data)  # type: ignore
        writer = PredictionWriter(
            outdir=predict["outdir"],
            datamodule=self.datamodule,
        )

        self.trainer = L.Trainer(callbacks=[writer], **trainer)

    def predict(self):
        self.trainer.predict(model=self.model, datamodule=self.datamodule)
