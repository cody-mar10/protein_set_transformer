from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal, NamedTuple, TypedDict, TypeVar, overload

import tables as tb
import torch
from more_itertools import all_equal
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from pst.data.graph import (
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_EDGE_STRATEGY,
    _SENTINEL_THRESHOLD,
    GenomeGraph,
)
from pst.data.utils import H5_FILE_COMPR_FILTERS, graph_sizes_to_index_pointer
from pst.typing import EdgeIndexStrategy, FilePath, GenomeGraphBatch

GraphT = TypeVar("GraphT", bound=Data)
# TODO: what about scaffold?
FeatureLevel = Literal["node", "graph", "protein", "genome"]
_SENTINEL_FRAGMENT_SIZE = -1
logger = logging.getLogger(__name__)


class ScaffoldFeatureFragmentedData(TypedDict):
    weights: torch.Tensor
    class_id: torch.Tensor
    scaffold_registry: list[RegisteredFeature]


class FragmentedData(ScaffoldFeatureFragmentedData):
    sizes: torch.Tensor
    ptr: torch.Tensor
    genome_label: torch.Tensor
    scaffold_label: torch.Tensor


class RegisteredFeature(NamedTuple):
    name: str
    data: torch.Tensor
    feature_level: FeatureLevel


class GenomeDataset(Dataset[GenomeGraphBatch]):
    __h5_fields__ = {
        "data",
        "ptr",
        "sizes",
        "class_id",
        "strand",
        "scaffold_label",
        "genome_label",
    }
    __optional_h5_fields__ = {"class_id", "scaffold_label", "genome_label"}
    __node_attr__ = {"x", "strand"}
    __graph_attr__ = {"edge_index", "size", "weight", "class_id"}

    ### protein level ###
    data: torch.Tensor
    strand: torch.Tensor

    ### scaffold level <- contiguous sequence of proteins ###
    ptr: torch.Tensor
    sizes: torch.Tensor
    class_id: torch.Tensor
    weights: torch.Tensor
    edge_indices: list[torch.Tensor]

    # shape: [N *scaffolds], since the ptr/sizes may refer to artificially chunked
    # scaffolds (.fragment). This should map subchunks to scaffolds. If .fragment was
    # not called, then this should be just be a tensor that ranges from 0..N scaffolds.
    # Otherwise, it will have repeated values to label each subchunk to its scaffold.
    scaffold_label: torch.Tensor

    ### genome level <- can be a single scaffold or multiple scaffolds ###
    # shape: [N *scaffolds], since the ptr/sizes may refer to artificially chunked
    # scaffolds (.fragment). This should map subchunks to genomes but also accounts
    # for naturally fragmented genomes or metagenomes.
    genome_label: torch.Tensor

    _FEATURE_MAP: dict[FeatureLevel, FeatureLevel] = {
        "protein": "node",
        "genome": "graph",
    }

    def __init__(
        self,
        file: FilePath,
        edge_strategy: EdgeIndexStrategy = _DEFAULT_EDGE_STRATEGY,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
        log_inverse: bool = False,
        fragment_size: int = _SENTINEL_FRAGMENT_SIZE,
    ) -> None:
        super().__init__()
        with tb.File(file) as fp:
            for field in GenomeDataset.__h5_fields__:
                try:
                    data = getattr(fp.root, field)
                except tb.exceptions.NoSuchNodeError:
                    if field in GenomeDataset.__optional_h5_fields__:
                        # the class_id field is not required for inference
                        # this was only used for weighting the loss
                        continue
                    else:
                        raise
                setattr(self, field, torch.from_numpy(data[:]))

        if not hasattr(self, "genome_label"):
            # assume each scaffold is a separate genome
            self.genome_label = torch.arange(len(self.sizes), dtype=torch.long)

        if not hasattr(self, "scaffold_label"):
            self.scaffold_label = self.genome_label.clone()

        # convert strand array from [-1, 1] -> [0, 1]
        # this will be used as an idx in a lut embedding
        self.strand[self.strand == -1] = 0

        self.weights = self._get_class_weights(log_inverse)

        if not hasattr(self, "class_id"):
            # default all genomes to the same class
            self.class_id = torch.zeros(len(self), dtype=torch.long)

        self._registered_features: dict[str, RegisteredFeature] = dict()

        # just keep track of names
        self._graph_registry: set[str] = set()
        self._node_registry: set[str] = set()

        self.edge_create_fn = GenomeGraph._edge_index_create_method(
            edge_strategy=edge_strategy, chunk_size=chunk_size, threshold=threshold
        )

        if fragment_size != _SENTINEL_FRAGMENT_SIZE:
            if fragment_size <= 1:
                raise ValueError(
                    f"Invalid fragment size {fragment_size}. Must be and integer greater than 1."
                )

            self.fragment(fragment_size, inplace=True)

        self.edge_indices = self.compute_edge_indices()

    def _get_class_weights(self, log_inverse: bool = True) -> torch.Tensor:
        if hasattr(self, "class_id"):
            # calc using inverse frequency
            # convert to ascending 0..n range
            class_counts: torch.Tensor
            _, inverse_index, class_counts = torch.unique(
                self.class_id, return_inverse=True, return_counts=True
            )
            freq: torch.Tensor = class_counts / class_counts.sum()
            inv_freq = 1.0 / freq
            if log_inverse:
                # with major class imbalance the contribution from rare classes can
                # be extremely high relative to other classes
                inv_freq = torch.log(inv_freq)

            # not sure if normalization does anything since all still contribute
            # the relative same amount to loss
            inv_freq /= torch.amin(inv_freq)

            # inverse index remaps input class_ids to 0..n range if not already
            weights = inv_freq[inverse_index]
        else:
            # no weights
            weights = torch.ones(size=(len(self),))

        return weights

    def compute_edge_indices(self) -> list[torch.Tensor]:
        # needed to compute edge indices for new fragmented datasets
        edge_indices: list[torch.Tensor] = list()

        for num_nodes in self.sizes:
            edge_index = self.edge_create_fn(num_nodes=num_nodes)
            edge_indices.append(edge_index)

        return edge_indices

    def __len__(self) -> int:
        return self.sizes.numel()

    @overload
    def __getitem__(self, idx: int) -> GenomeGraph: ...

    @overload
    def __getitem__(self, idx: slice) -> GenomeGraphBatch: ...

    @overload
    def __getitem__(self, idx: Iterable[int]) -> GenomeGraphBatch: ...

    def __getitem__(
        self, idx: int | slice | Iterable[int]
    ) -> GenomeGraph | GenomeGraphBatch:
        # this is scaffold-level access, for genome-level access use .get_genome
        if isinstance(idx, int):
            # idx should be for a single scaffold
            try:
                start = self.ptr[idx]
                stop = self.ptr[idx + 1]
            except IndexError as e:
                clsname = self.__class__.__name__
                raise IndexError(
                    f"{idx=} is out of range for {clsname} with {len(self)} genomes"
                ) from e

            registered_features: dict[str, torch.Tensor | int | float] = {}

            #### node/item level access (ptn)
            x = self.data[start:stop]
            strand = self.strand[start:stop]

            for name in self._node_registry:
                data = self._registered_features[name].data
                registered_features[name] = data[start:stop]

            #### graph/set level access (genome)
            edge_index = self.edge_indices[idx]
            num_proteins = int(self.sizes[idx])
            weight = self.weights[idx].item()
            class_id = int(self.class_id[idx])
            # shape: [N, 1]
            pos = torch.arange(num_proteins).unsqueeze(-1).to(x.device)
            scaffold_label = int(self.scaffold_label[idx])
            genome_label = int(self.genome_label[idx])

            for name in self._graph_registry:
                data = self._registered_features[name].data
                registered_features[name] = data[idx]

            # the edge index will already be created so no need to pass edge creation info
            graph = GenomeGraph(
                x=x,
                edge_index=edge_index,
                num_proteins=num_proteins,
                weight=weight,
                class_id=class_id,
                strand=strand,
                pos=pos,
                scaffold_label=scaffold_label,
                genome_label=genome_label,
                **registered_features,  # type: ignore
            )
            return graph
        else:
            if isinstance(idx, slice):
                idx = range(*idx.indices(len(self)))
            elif isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            elif isinstance(idx, Iterable):
                idx = list(idx)
            else:
                raise ValueError(f"Invalid index type {type(idx)}")
            return self.collate([self[i] for i in idx])

    def get_genome(self, genome_id: int) -> GenomeGraphBatch:
        # always return a batch obj even if only one genome
        # the batch obj has all the methods of the single graph obj

        genome_idx = torch.where(self.genome_label == genome_id)[0]
        genome_idx = genome_idx.tolist()
        return self[genome_idx]

    def _register_feature(
        self, name: str, data: torch.Tensor, feature_level: FeatureLevel
    ):
        # convert protein -> node, genome -> graph to get correct attribute
        feature_level = GenomeDataset._FEATURE_MAP.get(feature_level, feature_level)

        name_registry: set[str] = getattr(self, f"_{feature_level}_registry")
        name_registry.add(name)

        feature = RegisteredFeature(name=name, data=data, feature_level=feature_level)
        self._registered_features[name] = feature

    def register_feature(
        self,
        name: str,
        data: torch.Tensor,
        *,
        feature_level: FeatureLevel,
        overwrite_previously_registered: bool = False,
    ):
        if name in self.__node_attr__ or name in self.__graph_attr__:
            raise ValueError(
                f"Cannot register feature with name {name} as it is a reserved attribute."
            )
        elif name in self._registered_features:
            if not overwrite_previously_registered:
                raise ValueError(
                    f"Feature with name '{name}' has already been registered. Set "
                    "`overwrite_previously_registered=True` to overwrite."
                )
            # the next code will just overwrite it

        if feature_level == "node" or feature_level == "protein":
            if data.shape[0] != self.data.shape[0]:
                raise ValueError(
                    f"Expected {self.data.shape[0]} items in data, got {data.shape[0]}"
                )
            self._register_feature(name, data, feature_level)
        elif feature_level == "graph" or feature_level == "genome":
            if data.shape[0] != self.sizes.shape[0]:
                raise ValueError(
                    f"Expected {self.sizes.shape[0]} items in data, got {data.shape[0]}"
                )
            self._register_feature(name, data, feature_level)
        else:
            raise ValueError(
                (
                    "Invalid feature level. Must be one of 'node'/'protein' or 'graph'/'genome'"
                    "to indicate if the registered feature describes a node-level or "
                    "graph-level property, respectively."
                )
            )

    def get_registered_feature(self, name: str) -> torch.Tensor:
        return self._registered_features[name].data

    @property
    def feature_dim(self) -> int:
        return int(self.data.shape[-1])

    @property
    def max_size(self) -> int:
        return int(self.sizes.amax())

    @staticmethod
    def collate(batch: list[GenomeGraph]) -> GenomeGraphBatch:
        databatch: GenomeGraphBatch = Batch.from_data_list(batch)  # type: ignore

        # we need to keep the genome and scaffold labels RELATIVE
        # this is needed since the scatter-style reductions will create
        # empty slots in any reduction tensors that will not be of the correct size

        _, rel_scaffold_label = databatch.scaffold_label.unique(
            sorted=False, return_inverse=True
        )
        _, rel_genome_label = databatch.genome_label.unique(
            sorted=False, return_inverse=True
        )

        # scaffold_label and genome_label are not really relevant at the minibatch level
        # they are more useful at the dataset level where we can artificially fragment
        # genomes. In these scenarios, we can keep track of which fragments belong to which
        # scaffolds, and which scaffolds belong to which genomes (if we have metagenomes).
        # however, they're still provided at the minibatch level for convenience

        # scaffold_label also needs to be expanded to match the number of proteins
        # technically the pyg batching procedure computes this exactly
        # with the .batch attr, so really the .scaffold_label is only
        # important for the dataset level and not the minibatch level
        # shape: [N scaffolds] -> [N proteins]
        # actually this is only used for REDUCING OVER sub-scaffold CHUNKS that were artifically created
        # so needs to be shape: [N scaffolds]
        databatch.scaffold_label = rel_scaffold_label

        # we dont expand the genome_label tensor (shape: [N scaffolds])
        # this is really more useful for inference and not for model training
        # it is not really guaranteed to sample all scaffolds from a single genome
        # without a custom sampler
        # also the entire point of PST is context from a contiguous sequence of proteins
        # so users can use this after inference converts the data from
        # [N proteins, D] -> [N scaffolds, D]
        # to get genome level representations ([N genomes, D])
        databatch.genome_label = rel_genome_label

        return databatch

    def collate_indices(self, idx_batch: list[torch.Tensor]) -> GenomeGraphBatch:
        batch = [self[int(idx)] for idx in idx_batch]
        return self.collate(batch)

    def _resize_scaffold_features(
        self, chunk_scaffold_labels: torch.Tensor
    ) -> ScaffoldFeatureFragmentedData:
        # need to resize scaffold level features during fragmenting
        # just propagate labels to each sub-scaffold fragment
        features: ScaffoldFeatureFragmentedData = {
            "weights": self.weights[chunk_scaffold_labels],
            "class_id": self.class_id[chunk_scaffold_labels],
            "scaffold_registry": list(),
        }
        for scaffold_feature in self._graph_registry:
            feature = self._registered_features[scaffold_feature]
            new_feature = RegisteredFeature(
                name=feature.name,
                data=feature.data[chunk_scaffold_labels],
                feature_level=feature.feature_level,
            )
            features["scaffold_registry"].append(new_feature)

        return features

    def _fragment(self, max_size: int) -> FragmentedData:
        # basically all we do to artificially fragment scaffolds
        # break the .ptr and .sizes tensors into smaller chunks
        # then we keep track of which genome and scaffold each chunk belongs to
        # those label tensors can be used to group the chunks back together
        # thru averaging or summing the features

        # these are per genome
        num_max_size_chunks = torch.floor(self.sizes / max_size).long()
        size_of_smallest_chunk = self.sizes % max_size
        n_chunks_per_genome = num_max_size_chunks + (size_of_smallest_chunk > 0).long()

        total_chunks = int(n_chunks_per_genome.sum())

        # create a new tensors to hold the data

        # assign each back to its original genome
        chunk_genome_labels = torch.zeros(total_chunks, dtype=torch.long)
        chunk_sizes = chunk_genome_labels.clone()
        chunk_scaffold_labels = chunk_genome_labels.clone()

        chunk_pos = 0
        for scaffold_id, (
            genome_id,
            n_max_size_chunks,
            smallest_chunk_size,
            n_chunks,
        ) in enumerate(
            zip(
                self.genome_label,
                num_max_size_chunks,
                size_of_smallest_chunk,
                n_chunks_per_genome,
            )
        ):
            for i in range(n_chunks):
                if i < n_max_size_chunks:
                    chunk_size = max_size
                else:
                    chunk_size = smallest_chunk_size

                chunk_sizes[chunk_pos] = chunk_size
                chunk_genome_labels[chunk_pos] = genome_id
                chunk_scaffold_labels[chunk_pos] = scaffold_id
                chunk_pos += 1

        chunk_index_ptr = graph_sizes_to_index_pointer(chunk_sizes)

        scaffold_features = self._resize_scaffold_features(chunk_scaffold_labels)

        output = FragmentedData(
            sizes=chunk_sizes,
            ptr=chunk_index_ptr,
            genome_label=chunk_genome_labels,
            scaffold_label=chunk_scaffold_labels,
            **scaffold_features,
        )
        return output

    def _update_from_fragmented_data(self, fragmented_data: FragmentedData):
        scaffold_feature: RegisteredFeature
        for key, value in fragmented_data.items():
            if key == "scaffold_registry":
                for scaffold_feature in value:  # type: ignore
                    self.register_feature(
                        name=scaffold_feature.name,
                        data=scaffold_feature.data,
                        feature_level=scaffold_feature.feature_level,
                        overwrite_previously_registered=True,
                    )
            else:
                setattr(self, key, value)

        self.edge_indices = self.compute_edge_indices()
        self.validate()
        logger.debug(
            f"Fragmented dataset into {len(self)} fragments with <= "
            f"{self.max_size} proteins."
        )

    def fragment(self, max_size: int, inplace: bool = True):
        fragmented_data = self._fragment(max_size)

        if inplace:
            self._update_from_fragmented_data(fragmented_data)
            return self

        from copy import copy, deepcopy

        # this makes a shallow copy, so the underlying tensors are actually shared
        # but we can still reassign the tensors. Most signficant reason to do this
        # is to not have to copy the protein embeddings
        new_dataset = copy(self)
        # however, we need to make a deep copy of the registered features
        # so that the new and old datasets don't share the same dict
        new_dataset._registered_features = deepcopy(self._registered_features)
        new_dataset._update_from_fragmented_data(fragmented_data)

        return new_dataset

    def validate(self):
        ### check number of proteins
        num_protein_embeddings = self.data.shape[0]

        num_proteins_from_ptr = int(self.ptr[-1])
        if num_proteins_from_ptr != num_protein_embeddings:
            raise RuntimeError(
                f"Expected {num_protein_embeddings} proteins, but the index pointer reports {num_proteins_from_ptr}"
            )

        num_proteins_from_sizes = int(self.sizes.sum())
        if num_proteins_from_sizes != num_protein_embeddings:
            raise RuntimeError(
                f"Expected {num_protein_embeddings} proteins, but the sizes tensor reports {num_proteins_from_sizes}"
            )

        ### check number of scaffolds
        num_scaffolds = len(self)
        num_scaffolds_from_ptr = len(self.ptr) - 1
        num_scaffolds_from_edge_indices = len(self.edge_indices)
        num_scaffolds_from_genome_label = self.genome_label.numel()
        num_scaffolds_from_scaffold_label = self.scaffold_label.numel()
        num_scaffolds_from_weight = self.weights.numel()
        num_scaffolds_from_class_id = self.class_id.numel()

        num_scaffolds_from_registry = [
            self._registered_features[name].data.shape[0]
            for name in self._graph_registry
        ]

        if not all_equal(
            (
                num_scaffolds,
                num_scaffolds_from_ptr,
                num_scaffolds_from_edge_indices,
                num_scaffolds_from_genome_label,
                num_scaffolds_from_scaffold_label,
                num_scaffolds_from_weight,
                num_scaffolds_from_class_id,
                *num_scaffolds_from_registry,
            )
        ):
            raise RuntimeError(
                f"Expected {num_scaffolds} scaffolds but get the following number of scaffolds "
                f"from different internal data suggest:\nptr={num_scaffolds_from_ptr}\n"
                f"edge_indices={num_scaffolds_from_edge_indices}\n"
                f"genome_label={num_scaffolds_from_genome_label}\n"
                f"scaffold_label={num_scaffolds_from_scaffold_label}\n"
                f"weight={num_scaffolds_from_weight}\n"
                f"class_id={num_scaffolds_from_class_id}"
            )

    def save(self, file: FilePath):
        """Save the dataset to an HDF5 file."""
        with tb.open_file(file, "w") as fp:
            for field in GenomeDataset.__h5_fields__:
                data = getattr(self, field).numpy()
                fp.create_carray("/", field, data, filters=H5_FILE_COMPR_FILTERS)

    def any_genomes_have_multiple_scaffolds(self) -> bool:
        num_scaffolds = self.scaffold_label.max().item()
        num_genomes = self.genome_label.max().item()

        return num_scaffolds != num_genomes
