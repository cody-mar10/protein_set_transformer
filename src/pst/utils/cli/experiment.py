from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

AnnealingOpts = Literal["linear", "cos"]
ModelType = Literal[
    "ProteinSetTransformer", "MLMProteinSetTransformer", "ProteinSetTransformerEncoder"
]


class ExperimentArgs(BaseModel):
    # this has to stay here so it is always available even when not tuning
    config: Optional[Path] = Field(
        None,
        description="config toml file to specify which hyperparameters to tune with optuna",  # noqa: E501
    )
    name: str = Field(
        "exp0", description="experiment name in logging directory during training"
    )
    patience: int = Field(
        5, description="early stopping patience, see lightning description", gt=0
    )
    save_top_k: int = Field(1, description="save the best k models", gt=0)
    ### SWA args
    swa: bool = Field(False, description="use stochastic weight averaging")
    swa_epoch_start: int | float = Field(
        0.8,
        description=(
            "when to start SWA if enabled. Integers are an epoch number, "
            "while floats in range (0.0, 1.0) are a fraction of max epochs. "
        ),
    )
    annealing_epochs: int = Field(
        10,
        description="number of epochs to anneal learning rate during swa if enabled",
        gt=0,
    )
    annealing_strategy: AnnealingOpts = Field(
        "linear", description="annealing strategy using during swa if enabled"
    )
    ### related to tuning -> get params from previous runs
    from_study: Optional[Path] = Field(
        None, description="load hyperparameters from previous tuning study"
    )
    from_json: Optional[Path] = Field(
        None, description="load hyperparameters from json file"
    )
    log_gradients: bool = Field(False, description="debug mode to log gradients")
    detect_anomaly: bool = Field(
        False, description="debug mode to detect gradient anomalies"
    )
    pst_model_type: ModelType = Field(
        "ProteinSetTransformer",
        description=(
            "model type to train. `ProteinSetTransformer` is described in the manuscript and is "
            "trained with triplet loss and pointswap augmentation. `MLMProteinSetTransformer` "
            "is trained using masked language modeling loss with positive sampling. "
            "`ProteinSetTransformerEncoder` is an encoder-only variant of "
            "`ProteinSetTransformer` that trains only on protein embeddings with triplet loss "
            "at the protein level."
        ),
    )
    min_delta: float = Field(
        0.05,
        description=(
            "loss has to improve by this much before --patience epochs otherwise training will "
            "stop early"
        ),
        ge=0.0,
    )
