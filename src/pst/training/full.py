from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

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

    checkpoint_callback = ModelCheckpoint(
        save_last=True, save_top_k=config.experiment.save_top_k, every_n_epochs=1
    )

    # want max time value, not enum
    trainer = L.Trainer(
        max_time=config.trainer.max_time.value,
        callbacks=[checkpoint_callback],
        **config.trainer.model_dump(exclude={"max_time"}),
    )
    trainer.fit(model=model, datamodule=datamodule)
