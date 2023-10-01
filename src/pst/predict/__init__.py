from __future__ import annotations

from typing import Any

import lightning as L
import torch

from pst.arch.data import DataConfig, GenomeDataModule
from pst.arch.modules import ModelConfig
from pst.arch.modules import ProteinSetTransformer as PST
from pst.utils.cli.modes import InferenceMode

from .writer import PredictionWriter


def model_inference(config: InferenceMode, return_predictions: bool = False):
    ckpt: dict[str, Any] = torch.load(
        config.predict.checkpoint,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    model_config = ModelConfig.model_validate(ckpt["hyper_parameters"])

    model = PST(model_config)
    model.load_state_dict(ckpt["state_dict"])

    data_config = DataConfig.model_construct(
        file=config.data.file,
        **ckpt["datamodule_hyper_parameters"],
    )

    datamodule = GenomeDataModule(data_config)

    expected_max_size = model.positional_embedding.max_size
    actual_max_size = datamodule.dataset.max_size
    if actual_max_size > expected_max_size:
        # TODO: the default behavior for other transformers is to just work with the first
        # n words that work with the model's max size... that is sort of complicated here
        # since that would require adjusting the genome's graph representation

        raise RuntimeError(
            (
                f"The maximum number of proteins in your dataset is {actual_max_size}, "
                f"but this model was trained with a max of {expected_max_size} proteins."
            )
        )

    writer = PredictionWriter(
        outdir=config.predict.outdir,
        datamodule=datamodule,
    )

    trainer = L.Trainer(
        callbacks=[writer],
        max_time=config.trainer.max_time.value,
        **config.trainer.model_dump(exclude={"max_time"}),
    )

    outputs = trainer.predict(
        model=model, datamodule=datamodule, return_predictions=return_predictions
    )

    if return_predictions:
        return outputs
