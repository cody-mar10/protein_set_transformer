from typing import Literal

from attrs import define

ManuscriptChoices = Literal[
    "source_data",
    "supplementary_data",
    "supplementary_tables",
    "host_prediction",
    "fasta",
    "foldseek_databases",
    "README",
]


@define
class ManuscriptDataArgs:
    """MANUSCRIPT DATA"""

    choices: list[ManuscriptChoices] | None = None
    "Download manuscript-specific data. Defaults to only the README."

    def __attrs_post_init__(self):
        if self.choices is None:
            self.choices = ["README"]


ClusterChoices = Literal["genome", "protein"]


@define
class ClusterArgs:
    """CLUSTER DATA"""

    choices: list[ClusterChoices] | None = None
    """Download genome or protein clusters."""


ModelChoices = Literal[
    "PST-TL-P__small",
    "PST-TL-P__large",
    "PST-TL-T__small",
    "PST-TL-T__large",
    "PST-MLM",
]


@define
class ModelArgs:
    """PRETRAINED MODELS"""

    choices: list[ModelChoices] | None = None
    """Download pretrained models."""


EmbeddingChoices = Literal[
    ### protein
    "esm2",
    "IMGVR_PST-TL-P__large",
    "IMGVR_PST-TL-P__small",
    "IMGVR_PST-TL-T__large",
    "IMGVR_PST-TL-T__small",
    "MGnify_PST-TL-P__large",
    "MGnify_PST-TL-P__small",
    "MGnify_PST-TL-T__large",
    "MGnify_PST-TL-T__small",
    "genslm_ORF",
    "train_PST-TL-P__large",
    "train_PST-TL-P__small",
    "train_PST-TL-T__large",
    "train_PST-TL-T__small",
    ### genome
    "PST-TL_genome",
    "other_genome",
]


@define
class EmbeddingsArgs:
    """EMBEDDINGS"""

    choices: list[EmbeddingChoices] | None = None
    """Download embedding files."""
