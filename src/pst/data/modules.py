from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from lightning_cv import CrossValidationDataModule
from lightning_cv.split import ImbalancedLeaveOneGroupOut
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from pst.data.dataset import _SENTINEL_FRAGMENT_SIZE, FeatureLevel, GenomeDataset
from pst.typing import EdgeIndexStrategy


class DataConfig(BaseModel):
    file: Path = Field(
        ...,
        description=(
            "input protein embeddings file in `pst graphify` .h5 file format. See the wiki for "
            "more information if manually creating this file. Otherwise, the `pst graphify`"
            "workflow will create this file correctly."
        ),
    )
    batch_size: int = Field(
        32, description="batch size in number of genomes", le=128, multiple_of=2
    )
    train_on_full: bool = Field(
        False,
        description=(
            "whether to train a single model on the full input data. When not specified, the "
            "default is to train multiple models with cross validation"
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
            "size of sub-chunks to break genomes into if using --edge_strategy chunked. This is "
            "the range of protein-protein neighborhoods within a contiguous scaffold. This is "
            "different from --fragment-size, which controls artificially fragmenting scaffolds "
            "before protein-protein neighborhoods are calculated. Protein-protein neighborhoods "
            "are only calculated within a contiguous scaffold (happens AFTER --fragment-size "
            "fragmenting)"
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
    fragment_size: int = Field(
        _SENTINEL_FRAGMENT_SIZE,
        description=(
            "artificially break scaffolds into fragments that have no more than this many "
            "proteins. This is different from --chunk-size, which controls the range of "
            "protein-protein neighborhoods within a contiguous scaffold. This value simulates "
            "smaller scaffolds before protein-protein neighborhoods are calculated. Setting "
            "this value can reduce the memory burden, especially for large genomes like bacteria "
            "that encode thousands of proteins. Default is -1, which means NO fragmentation."
            "During INFERENCE or FINETUNING ONLY, if this value is not -1, the dataset will be "
            "be automatically fragmented into fragments of this --max-proteins size (if "
            "--fragment-oversized-genomes used), which may still use too much memory, so consider "
            "setting this to be smaller than --max-proteins"
        ),
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
        "fragment_size",
    }

    def __init__(self, config: DataConfig, **kwargs) -> None:
        self.config = config
        self.dataset = GenomeDataset(
            file=config.file,
            edge_strategy=config.edge_strategy,
            chunk_size=config.chunk_size,
            threshold=config.threshold,
            log_inverse=config.log_inverse,
            fragment_size=config.fragment_size,
        )
        super().__init__(
            dataset=self.dataset,
            batch_size=config.batch_size,
            cross_validator=ImbalancedLeaveOneGroupOut,  # TODO: should make this configurable
            cross_validator_config={"groups": self.dataset.scaffold_class_id},
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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        data_file: str | Path,
        command_line_config: Optional[DataConfig] = None,
        **kwargs,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = DataConfig.model_construct(
            file=Path(data_file), **ckpt["datamodule_hyper_parameters"]
        )

        # allow users to update config from command line
        # this is most significant for batch size and fragment size
        # edge creation strategy should have been tuned and shouldn't be updated
        if command_line_config is not None:
            ALLOWED_KEYS = {"batch_size", "fragment_size"}
            for key in ALLOWED_KEYS:
                # check if they were set from cli
                if key in command_line_config.model_fields_set:
                    # if so, update the config
                    setattr(config, key, getattr(command_line_config, key))

        return cls(config, **kwargs)

    def register_feature(
        self,
        name: str,
        data: torch.Tensor,
        *,
        feature_level: FeatureLevel,
    ):
        self.dataset.register_feature(name, data, feature_level=feature_level)
