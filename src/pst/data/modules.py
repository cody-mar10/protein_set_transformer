from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from lightning_cv import CrossValidationDataModule
from lightning_cv.split import ImbalancedLeaveOneGroupOut
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from pst.data.dataset import GenomeDataset
from pst.typing import EdgeIndexStrategy


class DataConfig(BaseModel):
    file: Path = Field(
        ...,
        description=(
            "input protein embeddings file in .h5 file format with the fields .data for "
            "protein embeddings"  # TODO: add more desc
        ),
    )
    batch_size: int = Field(
        32, description="batch size in number of genomes", gt=4, le=128, multiple_of=2
    )
    train_on_full: bool = Field(
        False,
        description=(
            "whether to train a single model on the full input data. When not "
            "specified, the default is  to train multiple models with cross "
            "validation"
        ),
    )
    pin_memory: bool = Field(
        True, description="whether to pin memory onto a CUDA GPU or not"
    )
    num_workers: int = Field(0, description="additional cpu workers to load data", ge=0)
    edge_strategy: EdgeIndexStrategy = Field(
        "chunked",
        description=(
            "strategy to create 'edges' between protein nodes in a genome graph. "
            "chunked = split genomes in --chunk-size chunks. sparse = remove "
            "interactions longer than --threshold. full = fully connected graph "
            "like a regular transformer."
        ),
    )
    chunk_size: int = Field(
        30,
        description=(
            "size of sub-chunks to break genomes into if using --edge_strategy chunked"
        ),
        ge=15,
        le=50,
    )
    threshold: int = Field(
        -1,
        description=(
            "range of protein interactions if using --edge_strategy [chunked|sparse]"
        ),
    )
    log_inverse: bool = Field(
        False, description="take the log of inverse class freqs as weights"
    )


_StageType = Literal["fit", "test", "predict"]


class GenomeDataModule(CrossValidationDataModule):
    _LOGGABLE_HPARAMS = {
        "batch_size",
        "edge_strategy",
        "chunk_size",
        "threshold",
        "log_inverse",
        "train_on_full",
    }

    def __init__(self, config: DataConfig, **kwargs) -> None:
        self.config = config
        self.dataset = GenomeDataset(
            file=config.file,
            edge_strategy=config.edge_strategy,
            chunk_size=config.chunk_size,
            threshold=config.threshold,
            log_inverse=config.log_inverse,
        )
        super().__init__(
            dataset=self.dataset,
            batch_size=config.batch_size,
            cross_validator=ImbalancedLeaveOneGroupOut,
            cross_validator_config={"groups": self.dataset.class_id},
            collate_fn=self.dataset.collate_indices,
            **kwargs,
        )

        # shared / global dataloader kwargs, but they can still be updated
        self.dataloader_kwargs = kwargs

        self.save_hyperparameters(config.model_dump(include=self._LOGGABLE_HPARAMS))

    def setup(self, stage: _StageType):
        if stage == "fit":
            if self.config.train_on_full:
                # train final model with all data
                self.train_dataset = self.dataset
            else:
                # train with cross validation
                # sets self.data_manager module which keeps track of the folds
                super().setup("fit")
        elif stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def simple_dataloader(self, dataset: GenomeDataset, **kwargs) -> DataLoader:
        kwargs = self._overwrite_dataloader_kwargs(**kwargs)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            **kwargs,
        )
        return dataloader

    def train_dataloader(self, **kwargs) -> DataLoader:
        return self.simple_dataloader(self.train_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return self.simple_dataloader(self.test_dataset, **kwargs)

    def predict_dataloader(self, **kwargs) -> DataLoader:
        return self.simple_dataloader(self.predict_dataset, **kwargs)

    # TODO: need to add a way to change kwargs like train_on_full
    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str | Path, data_file: str | Path, **kwargs
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = DataConfig.model_construct(
            file=Path(data_file), **ckpt["datamodule_hyper_parameters"]
        )

        return cls(config, **kwargs)
