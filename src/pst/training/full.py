from __future__ import annotations

import lightning as L

from pst.arch.data import GenomeDataModule
from pst.arch.modules import ProteinSetTransformer as PST
from pst.utils.cli.modes import TrainingMode


def train_with_all_data(config: TrainingMode):
    if not config.data.train_on_full:
        raise RuntimeError(
            "Cannot train with all data if --train-on-full not passed at command line."
        )

    datamodule = GenomeDataModule(config.data)
    # update model's in_dim
    if config.model.in_dim == -1:
        config.model.in_dim = datamodule.dataset.feature_dim

    model = PST(config.model)

    trainer = L.Trainer(**config.trainer.model_dump())
    trainer.fit(model=model, datamodule=datamodule)
