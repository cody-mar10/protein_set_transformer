from pathlib import Path
from typing import Generic, Literal, Optional, TypeVar, cast

import torch
from lightning import LightningDataModule
from lightning_cv import CrossValidationDataModuleMixin

from pst.data.config import CrossValDataConfig, CrossValidationType, CVStrategies, DataConfig
from pst.data.dataset import (
    FeatureLevel,
    GenomeDataset,
    LazyGenomeDataset,
    SubsetGenomeDataset,
    _BaseGenomeDataset,
)
from pst.data.loader import EmptyDataLoader, GenomeDataLoader
from pst.data.split import random_split
from pst.typing import KwargType
from pst.utils._signatures import _resolve_config_type_from_init

_StageType = Literal["fit", "test", "predict"]
_BaseConfigType = TypeVar("_BaseConfigType", bound=DataConfig)


class GenomeDataModuleMixin(LightningDataModule, Generic[_BaseConfigType]):
    _LOGGABLE_HPARAMS = {
        "batch_size",
        "edge_strategy",
        "chunk_size",
        "threshold",
        "log_inverse",
        "fragment_size",
        "dataloader",
    }

    config: _BaseConfigType
    dataset: _BaseGenomeDataset
    train_dataset: _BaseGenomeDataset | SubsetGenomeDataset
    val_dataset: Optional[_BaseGenomeDataset | SubsetGenomeDataset]
    test_dataset: _BaseGenomeDataset
    predict_dataset: _BaseGenomeDataset

    def __init__(
        self,
        config: _BaseConfigType,
        extra_save_hyperparameters: Optional[set[str]] = None,
        **dataloader_kwargs,
    ):
        if self._is_base_class():
            raise TypeError(
                "GenomeDataModuleMixin should not be instantiated directly. Use a concrete "
                "subclass."
            )

        expected_config_type = self._resolve_config_type()
        if not isinstance(config, expected_config_type):
            raise TypeError(f"Expected config of type {expected_config_type}, got {type(config)}")

        super().__init__()

        dataset_cls = LazyGenomeDataset if config.lazy else GenomeDataset

        self.dataset = dataset_cls(**config.to_dict(include=LazyGenomeDataset._init_arg_names()))
        self.config = config
        self.batch_size = config.batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self._dataloader = self.config.dataloader.value

        if extra_save_hyperparameters is None:
            extra_save_hyperparameters = set()

        included_hparams = self._LOGGABLE_HPARAMS.union(extra_save_hyperparameters)

        # must convert enums to str to be able to write hparams
        # deserialization of strs back to their enum members is handled automatically
        # by attrs
        self.save_hyperparameters(
            self.config.to_dict(include=included_hparams, convert_enum_to_str=True)
        )

    def _overwrite_dataloader_kwargs(self, **new_dataloader_kwargs):
        return self.dataloader_kwargs | new_dataloader_kwargs

    def register_feature(
        self,
        name: str,
        data: torch.Tensor,
        *,
        feature_level: FeatureLevel,
    ):
        self.dataset.register_feature(name, data, feature_level=feature_level)

    @classmethod
    def _resolve_config_type(cls) -> type[_BaseConfigType]:
        data_config_type = _resolve_config_type_from_init(
            cls, config_name="config", default=DataConfig
        )

        return cast(type[_BaseConfigType], data_config_type)

    def _is_base_class(self) -> bool:
        return self.__class__.__name__ == GenomeDataModuleMixin.__name__

    #### lightning datamodule methods
    def setup(self, stage: _StageType):
        if stage == "fit":
            if self.config.validation is None:
                self.train_dataset = self.dataset
                self.val_dataset = None
            elif self.config.validation == "random":
                split_kwargs: KwargType = dict(dataset=self.dataset, lengths=[0.8, 0.2])
                # 80:20 split
                if self._dataloader is GenomeDataLoader:
                    split_kwargs["split_level"] = "genome"
                else:
                    split_kwargs["split_level"] = "scaffold"

                # this will handle if we are splitting on scaffolds or genomes
                self.train_dataset, self.val_dataset = random_split(**split_kwargs)
            else:
                self.train_dataset = self.dataset

                # must be a path
                init_kwargs = self.config.to_dict(include=GenomeDataset._init_arg_names())

                del init_kwargs["file"]

                self.val_dataset = GenomeDataset(file=self.config.validation, **init_kwargs)
        elif stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def teardown(self, stage: str) -> None:
        if self.dataset.lazy:
            self.dataset._file.close()

    def simple_dataloader(self, dataset: _BaseGenomeDataset | SubsetGenomeDataset, **kwargs):
        kwargs = self._overwrite_dataloader_kwargs(**kwargs)
        kwargs["dataset"] = dataset
        if self._dataloader is not GenomeDataLoader:
            kwargs["collate_fn"] = _BaseGenomeDataset.collate
        kwargs["batch_size"] = self.batch_size
        return self._dataloader(**kwargs)

    def train_dataloader(self, **kwargs):
        return self.simple_dataloader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        if self.val_dataset is not None:
            kwargs["shuffle"] = False
            return self.simple_dataloader(self.val_dataset, **kwargs)

        # val loop will be a no op
        return EmptyDataLoader()

    def test_dataloader(self, **kwargs):
        return self.simple_dataloader(self.test_dataset, **kwargs)

    def predict_dataloader(self, **kwargs):
        return self.simple_dataloader(self.predict_dataset, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        data_file: str | Path,
        batch_size: Optional[int] = None,
        fragment_size: Optional[int] = None,
        lazy: bool = False,
        **kwargs,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        hparams: KwargType = ckpt["datamodule_hyper_parameters"]
        hparams["file"] = data_file
        config_type = cls._resolve_config_type()
        config = config_type.from_dict(hparams)

        # allow these to be changed irrespective of what is in the checkpoint
        # so these can be passed from the command line
        if batch_size is not None:
            config.batch_size = batch_size

        if fragment_size is not None:
            config.fragment_size = fragment_size

        config.lazy = lazy

        return cls(config, **kwargs)

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        return f"{clsname}(config={self.config})"

    def summarize(self) -> str:
        num_proteins = self.dataset.num_proteins
        num_genomes = self.dataset.num_genomes
        num_scaffolds = self.dataset.num_scaffolds
        num_features = len(self.dataset._registered_features)
        embedding_dim = self.dataset.feature_dim

        clsname = self.__class__.__name__
        summary = [
            f"{clsname}:",
            f"  - protein embeddings: ({num_proteins}, {embedding_dim})",
            f"  - {num_genomes=}",
            f"  - {num_scaffolds=}",
            f"  - {num_features=}",
        ]

        return "\n".join(summary)


## THIS IS WITHOUT CV
class GenomeDataModule(GenomeDataModuleMixin[DataConfig]):
    def __init__(self, config: DataConfig, **dataloader_kwargs):
        super().__init__(config, **dataloader_kwargs)


### THIS IS WITH CV
class _CrossValGenomeDataModule(
    GenomeDataModuleMixin[CrossValDataConfig],
):
    def __init__(self, config: CrossValDataConfig, **dataloader_kwargs):
        super().__init__(
            config,
            extra_save_hyperparameters={"cv_strategy", "cv_type", "cv_var_name"},
            **dataloader_kwargs,
        )


class CrossValGenomeDataModule(CrossValidationDataModuleMixin, _CrossValGenomeDataModule):
    def __init__(self, config: CrossValDataConfig, **dataloader_kwargs):
        _CrossValGenomeDataModule.__init__(self, config, **dataloader_kwargs)

        cross_validator_config, dataset_size, group_attr_name, label_attr_name = (
            self._setup_cross_validation()
        )

        CrossValidationDataModuleMixin.__init__(
            self,
            dataset=self.dataset,
            batch_size=self.batch_size,
            cross_validator=self.config.cv_strategy.value,
            cross_validator_config=cross_validator_config,
            dataset_size=dataset_size,
            dataloader_type=self.config.dataloader.value,
            y_attr=label_attr_name,
            group_attr=group_attr_name,
            **self.dataloader_kwargs,
        )

    def setup(self, stage: _StageType):
        if stage != "fit":
            raise RuntimeError("CrossValGenomeDataModule only supports 'fit' stage")

        # there is no need for any additional setup since the cross validation splitter
        # has already been setup during initialization

    def _get_group_data(self, group_attr_name: str) -> tuple[str, torch.Tensor]:
        # this is for backwards compatibility where `class_id`
        # in the original training datasets was used as the grouping var
        backup_name = "class_id"
        if group_attr_name not in self.dataset._registered_features:
            if backup_name not in self.dataset._registered_features:
                raise ValueError(
                    f"Group attribute {group_attr_name} not found in dataset features: "
                    f"{self.dataset._registered_features.keys()}. This should be registered in "
                    "the graph-formatted .h5 data file as a feature. If you do not need cross "
                    "validation, then you should use the `GenomeDataModule` instead."
                )
            group_attr_name = backup_name

        return group_attr_name, self.dataset.get_registered_feature(group_attr_name)

    def _get_label_data(self, label_attr_name: str) -> torch.Tensor:
        if label_attr_name not in self.dataset._registered_features:
            raise ValueError(
                f"Label attribute {label_attr_name} not found in dataset features: "
                f"{self.dataset._registered_features.keys()}. This should be registered in the "
                "graph-formatted .h5 data file as a feature."
            )

        return self.dataset.get_registered_feature(label_attr_name)

    def _sanitize_cv_var_name(
        self,
    ):
        if self.config.cv_type == CrossValidationType.both:
            if "," not in self.config.cv_var_name or self.config.cv_var_name.count(",") != 1:
                raise ValueError(
                    f"Expected a comma-separated list of label,group for cv_var_name when using "
                    f"`both` cv_type, got {self.config.cv_var_name}"
                )
        elif "," in self.config.cv_var_name:
            raise ValueError(
                "A ',' was detected in the cv_var_name, indicating that the cv split is "
                "happening on both a target label AND a group label. If that is the case, "
                f"then you should use the `both` cv_type instead of `{self.config.cv_type.value}`."
            )

    def _check_cv_strategy_and_val_type(self):
        cv_type = self.config.cv_type.value
        expected_cv_strategies: set[CVStrategies] = getattr(CVStrategies, f"{cv_type}_methods")()

        if self.config.cv_strategy not in expected_cv_strategies:
            raise ValueError(
                f"Cross validation strategy {self.config.cv_strategy} is not compatible with "
                f"cross validation type {cv_type}. Expected one of: {expected_cv_strategies}"
            )

    def _setup_cross_validation(self):
        self._check_cv_strategy_and_val_type()
        cross_validator_config: KwargType = dict()

        # TODO: what if these need to be programmatically set? not sure what use case that would be
        if self.config.cv_type == CrossValidationType.group:
            group_attr_name, group_id = self._get_group_data(self.config.cv_var_name)
            label_attr_name = None

            if self.config.cv_strategy == CVStrategies.ImbalancedLeaveOneGroupOut:
                cross_validator_config["groups"] = group_id
            elif self.config.cv_strategy == CVStrategies.GroupKFold:
                cross_validator_config["n_splits"] = int(group_id.unique().numel())

        elif self.config.cv_type == CrossValidationType.label:
            label_attr_name = self.config.cv_var_name
            group_attr_name = None
            label_id = self._get_label_data(self.config.cv_var_name)
            # only allowed cv strategy is StratifiedKFold
            cross_validator_config["n_splits"] = int(label_id.unique().numel())
        else:
            label_attr_name, group_attr_name = self.config.cv_var_name.split(",")
            group_attr_name, group_id = self._get_group_data(group_attr_name)
            label_id = self._get_label_data(label_attr_name)

            # only allowed cv strategy is StratifiedGroupKFold
            # we are going to split on the groups but try to keep label props balanced
            cross_validator_config["n_splits"] = int(group_id.unique().numel())

        if self.config.dataloader is GenomeDataLoader:
            dataset_size = self.dataset.num_genomes
        else:
            dataset_size = self.dataset.num_scaffolds

        return cross_validator_config, dataset_size, group_attr_name, label_attr_name
