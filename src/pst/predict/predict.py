from __future__ import annotations

from typing import Any

import lightning as L

from pst.arch import GenomeDataModule, ProteinSetTransformer
from pst.utils.cli import Args
from .writer import PredictionWriter


# TODO: no need for predict kwargs -> change to actual values,
class Predictor:
    def __init__(
        self,
        predict_kwargs: dict[str, Any],
        data_kwargs: dict[str, Any],
        trainer_kwargs: dict[str, Any],
    ):
        self.model = ProteinSetTransformer.load_from_checkpoint(
            predict_kwargs["checkpoint"]
        )
        self.datamodule = GenomeDataModule(shuffle=False, **data_kwargs)
        writer = PredictionWriter(
            outdir=predict_kwargs["outdir"],
            datamodule=self.datamodule,
        )

        self.trainer = L.Trainer(callbacks=[writer], **trainer_kwargs)

    @classmethod
    def from_cli_args(cls, args: Args):
        return cls(
            data_kwargs=args.data,
            trainer_kwargs=args.trainer,
            predict_kwargs=args.predict,
        )

    def predict(self):
        self.trainer.predict(model=self.model, datamodule=self.datamodule)
