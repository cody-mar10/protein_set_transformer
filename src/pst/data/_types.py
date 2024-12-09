from typing import Literal

from torch import Tensor


class ProteinFeaturesTypeMixin:
    # shapes should all refer to number of proteins
    protein_data: Tensor
    protein_strand: Tensor


class ScaffoldFeaturesTypeMixin:
    # shapes should all refer to number of scaffolds
    scaffold_ptr: Tensor
    scaffold_sizes: Tensor
    scaffold_edge_indices: list[Tensor]
    scaffold_label: Tensor
    scaffold_genome_label: Tensor
    scaffold_part_of_multiscaffold: Tensor


class GenomeFeaturesTypeMixin:
    # shapes should all refer to number of genomes
    genome_is_multiscaffold: Tensor


FeatureLevel = Literal["node", "graph", "protein", "scaffold", "genome"]
