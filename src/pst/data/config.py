from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from warnings import warn

from jsonargparse.typing import NonNegativeInt, PositiveInt, restricted_number_type
from lightning_cv.split import ImbalancedLeaveOneGroupOut
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from pst.data.dataset import _SENTINEL_FRAGMENT_SIZE
from pst.data.loader import GenomeDataLoader, ScaffoldDataLoader
from pst.typing import EdgeIndexStrategy
from pst.utils.dataclass_utils import DataclassValidatorMixin, validated_field


class CrossValidationType(str, Enum):
    label = "label"
    """splitting on a prediction label, ie y"""

    group = "group"
    """splitting on a group attribute, ie group_id"""

    both = "both"
    """splitting on both a prediction label and a group attribute"""


class DataLoaderType(Enum):
    genome = GenomeDataLoader
    scaffold = ScaffoldDataLoader


class CVStrategies(Enum):
    ImbalancedLeaveOneGroupOut = ImbalancedLeaveOneGroupOut
    GroupKFold = GroupKFold
    StratifiedKFold = StratifiedKFold
    StratifiedGroupKFold = StratifiedGroupKFold
    LeaveOneGroupOut = LeaveOneGroupOut

    @classmethod
    def group_methods(cls) -> set["CVStrategies"]:
        return {cls.ImbalancedLeaveOneGroupOut, cls.GroupKFold, cls.LeaveOneGroupOut}

    @classmethod
    def label_methods(cls) -> set["CVStrategies"]:
        return {cls.StratifiedKFold}

    @classmethod
    def both_methods(cls) -> set["CVStrategies"]:
        return {cls.StratifiedGroupKFold}


ChunkSize = restricted_number_type(
    "ChunkSize",
    int,
    [(">=", 10), ("<=", 50)],
    docstring="Chunk size must be in [10, 50]",
)

OptionalPositiveInt = restricted_number_type(
    "OptionalPositiveInt",
    int,
    [(">", 0), ("==", -1)],
    join="or",
    docstring="Positive integer or -1 (representing optional value)",
)


@dataclass
class DataConfig(DataclassValidatorMixin):
    """DATA

    Data configuration to load a GenomeDataModule that does not implement cross validation.

    Attributes:
        file (Path): input protein embeddings file in `pst graphify` .h5 file format. See the
            wiki for more information if manually creating this file. Otherwise, the `pst
            graphify` workflow will create this file correctly.
        validation (Path | Literal["random"] | None): How to form a validation set if specified.
            If not provided, no validation will occur, so
            when training a model, the model will be trained on the entire dataset provided to
            --file. If the input is a file path, this will be loaded as an independent dataset
            for validation. Otherwise, the indicated validation strategy will be used to split
            the data from the input file.
        batch_size (int): batch size in number of genomes.
        pin_memory (bool): whether to pin memory onto a CUDA GPU or not.
        num_workers (int): additional cpu workers to load data.
        edge_strategy (EdgeIndexStrategy): strategy to create 'edges' between protein nodes in a
            genome graph. chunked = split genomes in --chunk-size chunks. sparse = remove
            interactions longer than --threshold. full = fully connected graph like a regular
            transformer.
        chunk_size (int): size of sub-chunks to break genomes into if using --edge_strategy
            chunked. This is the range of protein-protein neighborhoods within a contiguous
            scaffold. This is different from --fragment-size, which controls artificially
            fragmenting scaffolds before protein-protein neighborhoods are calculated.
            Protein-protein neighborhoods are only calculated within a contiguous scaffold
            (happens AFTER --fragment-size fragmenting).
        threshold (int): range of protein interactions if using --edge_strategy [chunked|sparse].
            Default is for no thresholding to occur, meaning that all proteins in the same
            scaffold/chunk will be connected in the genome graph.
        log_inverse (bool): take the log of inverse class freqs as weights.
        fragment_size (int): artificially break scaffolds into fragments that have no more than
            this many proteins. Default is no fragmentation.
        dataloader (DataLoaderType): dataloader type for loading minibatches of data.
    """

    file: Path
    """input protein embeddings file in `pst graphify` .h5 file format. See the wiki for more 
    information if manually creating this file. Otherwise, the `pst graphify` workflow will 
    create this file correctly."""

    validation: Optional[Literal["random"] | Path] = None
    """How to form a validation set if specified. If not provided, no validation will occur, so
    when training a model, the model will be trained on the entire dataset provided to --file.
    If the input is a file path, this will be loaded as an independent dataset for validation.
    Otherwise, the input dataset to --file will be randomly split at an 80:20 ratio"""

    batch_size: int = validated_field(default=32, validator=PositiveInt)
    """batch size in number of genomes"""

    pin_memory: bool = True
    """whether to pin memory onto a CUDA GPU or not"""

    num_workers: int = validated_field(default=0, validator=NonNegativeInt)
    """additional cpu workers to load data"""

    edge_strategy: EdgeIndexStrategy = "chunked"
    """strategy to create 'edges' between protein nodes in a genome graph. chunked = split 
    genomes in --chunk-size chunks. sparse = remove interactions longer than --threshold. full 
    = fully connected graph like a regular transformer."""

    chunk_size: int = validated_field(default=30, validator=ChunkSize)
    """size of sub-chunks to break genomes into if using --edge_strategy chunked. This is the 
    range of protein-protein neighborhoods within a contiguous scaffold. This is different from
    --fragment-size, which controls artificially fragmenting scaffolds before protein-protein 
    neighborhoods are calculated. Protein-protein neighborhoods are only calculated within a 
    contiguous scaffold (happens AFTER --fragment-size fragmenting)"""

    threshold: int = validated_field(default=-1, validator=OptionalPositiveInt)
    """range of protein interactions if using --edge_strategy [chunked|sparse]. Default is
    for no thresholding to occur, meaning that all proteins in the same scaffold/chunk will
    be connected in the genome graph.
    """

    log_inverse: bool = False
    """take the log of inverse class freqs as weights"""

    fragment_size: int = validated_field(
        default=_SENTINEL_FRAGMENT_SIZE, validator=OptionalPositiveInt
    )
    """artificially break scaffolds into fragments that have no more than this many proteins
    Default is no fragmentation."""

    dataloader: DataLoaderType = DataLoaderType.scaffold
    """dataloader type for loading minibatches of data
    - `genome` will load the proteins from all scaffolds that belong to each genome, which is
    useful when there are multiscaffold genomes
    - `scaffold` will only load the proteins from each scaffold WITHOUT considering if the
    source genome has other scaffolds
    """

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.file, Path):
            self.file = Path(self.file)

        if not self.file.exists():
            raise FileNotFoundError(f"File not found: {self.file}")

        if self.validation is not None and self.validation != "random":
            if not isinstance(self.validation, Path):
                self.validation = Path(self.validation)

            if not self.validation.exists():
                raise FileNotFoundError(f"Validation file not found: {self.validation}")


@dataclass
class CrossValDataConfig(DataConfig):
    """CROSS VAL DATA

    Data configuration to load a CrossValGenomeDataModule that implements cross validation.

    Attributes:
        validation (Path | Literal["random"] | None): NOT USED for this mode. This mode will
            always train on multiple models with the indicated cross-validation strategy.
        cv_strategy (CVStrategies): cross-validation strategy.
        cv_type (CrossValidationType): what kind of label to split on for cross-validation.
        cv_var_name (str): name of the variable to split on for cross-validation. This should
            be registered as a feature in the .h5 file.
    """

    cv_strategy: CVStrategies = CVStrategies.ImbalancedLeaveOneGroupOut
    """cross-validation strategy"""

    cv_type: CrossValidationType = CrossValidationType.group
    """what kind of label to split on for cross-validation"""

    cv_var_name: str = "group_id"
    """name of the variable to split on for cross-validation. This should be registered as a
    feature in the .h5 file. If using --cv_type both, this should be a comma-separated list
    in the order of label,group. Ex: --cv_type both --cv_var_name 'y,group_id'"""

    def __post_init__(self):
        super().__post_init__()

        if self.validation is not None:
            warn(
                "validation is not used for cross-validation data modules",
                UserWarning,
            )

            # just in case
            self.validation = None
