from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import cached_property
from typing import Any, overload

import tables as tb
import torch
from more_itertools import all_equal
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from pst.data._types import (
    GenomeFeaturesTypeMixin,
    ProteinFeaturesTypeMixin,
    ScaffoldFeaturesTypeMixin,
)
from pst.data.graph import (
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_EDGE_STRATEGY,
    _SENTINEL_THRESHOLD,
    GenomeGraph,
)
from pst.data.utils import (
    H5_FILE_COMPR_FILTERS,
    FeatureLevel,
    RegisteredFeature,
    _FragmentedData,
    _ScaffoldFeatureFragmentedData,
    graph_sizes_to_index_pointer,
)
from pst.typing import EdgeIndexStrategy, FilePath, GenomeGraphBatch

_SENTINEL_FRAGMENT_SIZE = -1
logger = logging.getLogger(__name__)


class GenomeDataset(
    Dataset[GenomeGraphBatch],
    ProteinFeaturesTypeMixin,
    ScaffoldFeaturesTypeMixin,
    GenomeFeaturesTypeMixin,
):
    # mixins are to provide obj attribute type annotations
    # the names of these attributes refer to what they are labeling and thus what the expected
    # shape of each tensor should be. For example, "protein_data" should be of shape:
    # [num proteins, D]. The __X_attr__ class attributes define the attributes at each level.

    __minimum_h5_fields__ = {"data", "ptr", "sizes", "strand"}
    __expected_h5_fields__ = __minimum_h5_fields__ | {"scaffold_label", "genome_label"}

    # the names of the object attributes at each level will be prefixed with the level name
    # if you want the protein embeddings, you can refer to `dataset.protein_data`
    __protein_attr__ = {"x", "data", "strand"}
    __scaffold_attr__ = {
        "ptr",
        "sizes",
        # "class_id",
        "scaffold_label",
        "genome_label",
        "edge_indices",
        "part_of_multiscaffold",
    }
    __genome_attr__ = {"is_multiscaffold"}

    __feature_levels__ = {"protein", "scaffold", "genome"}

    _FEATURE_MAP: dict[FeatureLevel, FeatureLevel] = {
        "node": "protein",
        "graph": "scaffold",
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

        self._setup_registry()
        self._read_h5_file(file)

        if not hasattr(self, "scaffold_genome_label"):
            # assume each scaffold is a separate genome
            self.scaffold_genome_label = torch.arange(
                len(self.scaffold_sizes), dtype=torch.long
            )

        if not hasattr(self, "scaffold_label"):
            self.scaffold_label = torch.arange(
                len(self.scaffold_sizes), dtype=torch.long
            )

        # shape: [Num genomes]
        self.genome_is_multiscaffold = torch.bincount(self.scaffold_genome_label) > 1

        # shape: [Num scaffolds]
        self.scaffold_part_of_multiscaffold = self.genome_is_multiscaffold[
            self.scaffold_genome_label
        ]

        if self.any_multi_scaffold_genomes():
            logger.info("Multi-scaffold genomes detected.")

        # convert strand array from [-1, 1] -> [0, 1]
        # this will be used as an idx in a lut embedding
        self.protein_strand[self.protein_strand == -1] = 0

        self.edge_create_fn = GenomeGraph._edge_index_create_method(
            edge_strategy=edge_strategy,
            chunk_size=chunk_size,
            threshold=threshold,
        )

        if fragment_size != _SENTINEL_FRAGMENT_SIZE:
            if fragment_size <= 1:
                raise ValueError(
                    f"Invalid fragment size {fragment_size}. Must be and integer greater than 1."
                )

            self.fragment(fragment_size, inplace=True)
        else:
            # .fragment will compute edge indices and call .validate
            self.scaffold_edge_indices = self.compute_edge_indices()
            self.validate()

    def _setup_registry(self):
        # just keep track of names
        self._genome_registry: set[str] = set()
        self._scaffold_registry: set[str] = set()
        self._protein_registry: set[str] = set()

        self._registered_features: dict[str, RegisteredFeature] = dict()

    def _read_h5_file(self, file: FilePath):
        features_to_register: list[dict[str, Any]] = []
        with tb.File(file) as fp:
            for node in fp.root:
                name: str = node._v_name

                # allow subgroups in h5 file, all data fields beneath them will be registered
                if isinstance(node, tb.Group):
                    if name not in GenomeDataset.__feature_levels__:
                        raise ValueError(
                            f"Unknown feature level {name}. Subgroups must be one of {GenomeDataset.__feature_levels__}"
                        )
                    # register all fields beneath this group
                    for subnode in node:
                        if not isinstance(subnode, tb.Array):
                            raise ValueError(
                                f"Only arrays are allowed underneath H5 groups. Found {subnode}"
                            )
                        subname = subnode._v_name
                        features_to_register.append(
                            {
                                "name": subname,
                                "data": torch.from_numpy(subnode[:]),
                                "feature_level": name,
                            }
                        )
                else:
                    if name in GenomeDataset.__expected_h5_fields__:
                        # directly set these as attributes
                        if name in GenomeDataset.__protein_attr__:
                            prefix = "protein"
                        elif name in GenomeDataset.__scaffold_attr__:
                            prefix = "scaffold"
                        # there are no expected genome fields
                        else:
                            assert False, "UNREACHABLE"

                        obj_attr_name = (
                            name if name == "scaffold_label" else f"{prefix}_{name}"
                        )
                        setattr(self, obj_attr_name, torch.from_numpy(node[:]))
                    else:
                        # then other top level fields will become a registered feature
                        # so self._setup_registry() should be called before this method
                        # these are REQUIRED to label the name with "protein_", "scaffold_", or "genome_"
                        feature_level = name.split("_", 1)[0]
                        if feature_level not in GenomeDataset.__feature_levels__:
                            if name == "class_id":
                                # backwards compatibility
                                feature_level = "scaffold"
                            else:
                                raise ValueError(
                                    f"Field {name} is not a recognized protein, scaffold, or genome attribute."
                                )

                        features_to_register.append(
                            {
                                "name": name,
                                "data": torch.from_numpy(node[:]),
                                "feature_level": feature_level,
                            }
                        )

        # now need to check that the minimum required fields are set
        missing: set[str] = set()
        for name in GenomeDataset.__minimum_h5_fields__:
            if name in self.__protein_attr__:
                prefix = "protein"
            elif name in self.__scaffold_attr__:
                prefix = "scaffold"

            obj_attr_name = name if name == "scaffold_label" else f"{prefix}_{name}"

            if not hasattr(self, obj_attr_name):
                missing.add(obj_attr_name)

        if missing:
            raise ValueError(
                f"Missing required fields: {missing}. These fields are required to create a GenomeDataset object."
            )

        for feature in features_to_register:
            self.register_feature(**feature)

    def compute_edge_indices(self) -> list[torch.Tensor]:
        # needed to compute edge indices for new fragmented datasets
        edge_indices: list[torch.Tensor] = list()

        for num_nodes, is_multi_scaffold in zip(
            self.scaffold_sizes, self.scaffold_part_of_multiscaffold
        ):
            is_multi_scaffold = bool(is_multi_scaffold)

            if num_nodes == 1:
                if is_multi_scaffold:
                    # only multiscaffold genomes are allowed to have scaffolds with 1 protein
                    edge_index = torch.tensor([[0, 0]]).t().contiguous()
                else:
                    raise RuntimeError(
                        "Failed to create edge index. This is because the scaffold has only 1 protein."
                    )
            else:

                edge_index = self.edge_create_fn(num_nodes=num_nodes)

            edge_indices.append(edge_index)

        return edge_indices

    def __len__(self) -> int:
        return self.scaffold_sizes.numel()

    @property
    def num_proteins(self) -> int:
        return int(self.protein_data.shape[0])

    @property
    def num_scaffolds(self) -> int:
        return len(self)

    @property
    def num_genomes(self) -> int:
        return self.genome_is_multiscaffold.numel()

    @cached_property
    def num_proteins_per_genome(self) -> torch.Tensor:
        return scatter(self.scaffold_sizes, self.scaffold_genome_label, reduce="sum")

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
                start = self.scaffold_ptr[idx]
                stop = self.scaffold_ptr[idx + 1]
            except IndexError as e:
                clsname = self.__class__.__name__
                raise IndexError(
                    f"{idx=} is out of range for {clsname} with {len(self)} genomes"
                ) from e

            registered_features: dict[str, torch.Tensor | int | float] = {}

            #### node/item level access (ptn)
            x = self.protein_data[start:stop]
            strand = self.protein_strand[start:stop]

            for name in self._protein_registry:
                data = self._registered_features[name].data
                registered_features[name] = data[start:stop]

            #### graph/set level access (scaffold)
            edge_index = self.scaffold_edge_indices[idx]
            num_proteins = int(self.scaffold_sizes[idx])
            # shape: [N, 1]
            pos = torch.arange(num_proteins).unsqueeze(-1).to(x.device)
            scaffold_label = int(self.scaffold_label[idx])
            genome_label = int(self.scaffold_genome_label[idx])

            for name in self._scaffold_registry:
                data = self._registered_features[name].data
                registered_features[name] = data[idx]

            # the edge index will already be created so no need to pass edge creation info
            graph = GenomeGraph(
                x=x,
                edge_index=edge_index,
                num_proteins=num_proteins,
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

        genome_idx = torch.where(self.scaffold_genome_label == genome_id)[0]
        genome_idx = genome_idx.tolist()
        return self[genome_idx]

    def _register_feature(
        self, name: str, data: torch.Tensor, feature_level: FeatureLevel
    ):
        # feature_level is one of {protein, scaffold, genome}

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
        if name in (
            self.__protein_attr__ | self.__scaffold_attr__ | self.__genome_attr__
        ):
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

        # standardize names:
        # node -> protein
        # graph -> scaffold
        # otherwise keep the same (including genome which needs to be explicit!)
        # only possible values are {protein, scaffold, genome}
        feature_level = self._FEATURE_MAP.get(feature_level, feature_level)
        num_data_points = data.shape[0]

        if feature_level == "protein":
            if num_data_points != self.num_proteins:
                raise ValueError(
                    f"Expected {self.num_proteins} items in protein-level data, got {num_data_points}"
                )
        elif feature_level == "scaffold":
            if num_data_points != self.num_scaffolds:
                raise ValueError(
                    f"Expected {self.num_scaffolds} items in scaffold-level data, got {num_data_points}"
                )
        elif feature_level == "genome":
            if num_data_points != self.num_genomes:
                raise ValueError(
                    f"Expected {self.num_genomes} items in genome-level data, got {num_data_points}"
                )
        else:
            raise ValueError(
                (
                    "Invalid feature level. Must be one of 'protein', 'scaffold', or 'genome' "
                    "to indicate what biological level the registered feature labels"
                )
            )

        self._register_feature(name, data, feature_level)

    def get_registered_feature(self, name: str) -> torch.Tensor:
        # TODO: should allow people to prefix names?
        return self._registered_features[name].data

    @property
    def feature_dim(self) -> int:
        return int(self.protein_data.shape[-1])

    @property
    def max_size(self) -> int:
        return int(self.scaffold_sizes.amax())

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

        # shape: [N scaffolds]
        databatch.scaffold_label = rel_scaffold_label

        # TODO: update naming?
        # shape: [N scaffolds]
        databatch.genome_label = rel_genome_label

        return databatch

    def collate_indices(self, idx_batch: list[torch.Tensor]) -> GenomeGraphBatch:
        batch = [self[int(idx)] for idx in idx_batch]
        return self.collate(batch)

    def _resize_scaffold_features(
        self, chunk_scaffold_labels: torch.Tensor
    ) -> _ScaffoldFeatureFragmentedData:
        # need to resize scaffold level features during fragmenting
        # just propagate labels to each sub-scaffold fragment
        features: _ScaffoldFeatureFragmentedData = {
            "scaffold_part_of_multiscaffold": self.scaffold_part_of_multiscaffold[
                chunk_scaffold_labels
            ],
            "scaffold_registry": list(),
        }
        for scaffold_feature in self._scaffold_registry:
            feature = self._registered_features[scaffold_feature]
            new_feature = RegisteredFeature(
                name=feature.name,
                data=feature.data[chunk_scaffold_labels],
                feature_level=feature.feature_level,
            )
            features["scaffold_registry"].append(new_feature)

        return features

    def _fragment(self, max_size: int) -> _FragmentedData:
        # basically all we do to artificially fragment scaffolds
        # break the .ptr and .sizes tensors into smaller chunks
        # then we keep track of which genome and scaffold each chunk belongs to
        # those label tensors can be used to group the chunks back together
        # thru averaging or summing the features

        # these are per genome
        num_max_size_chunks = torch.floor(self.scaffold_sizes / max_size).long()
        size_of_smallest_chunk = self.scaffold_sizes % max_size
        n_chunks_per_genome = num_max_size_chunks + (size_of_smallest_chunk > 0).long()

        total_chunks = int(n_chunks_per_genome.sum())

        # create a new tensors to hold the data

        # assign each back to its original genome
        chunk_scaffold_genome_labels = torch.zeros(total_chunks, dtype=torch.long)
        chunk_sizes = chunk_scaffold_genome_labels.clone()
        chunk_scaffold_labels = chunk_scaffold_genome_labels.clone()

        chunk_pos = 0
        for scaffold_id, (
            genome_id,
            n_max_size_chunks,
            smallest_chunk_size,
            n_chunks,
        ) in enumerate(
            zip(
                self.scaffold_genome_label,
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
                chunk_scaffold_genome_labels[chunk_pos] = genome_id
                chunk_scaffold_labels[chunk_pos] = scaffold_id
                chunk_pos += 1

        chunk_index_ptr = graph_sizes_to_index_pointer(chunk_sizes)

        scaffold_features = self._resize_scaffold_features(chunk_scaffold_labels)

        # for attributes that are per genome, these do NOT need to be resized
        # since we are only breaking up scaffolds

        output = _FragmentedData(
            scaffold_sizes=chunk_sizes,
            scaffold_ptr=chunk_index_ptr,
            scaffold_genome_label=chunk_scaffold_genome_labels,
            scaffold_label=chunk_scaffold_labels,
            **scaffold_features,
        )
        return output

    def _update_from_fragmented_data(self, fragmented_data: _FragmentedData):
        registered_feature: RegisteredFeature
        for key, value in fragmented_data.items():
            if "_registry" in key:
                for registered_feature in value:  # type: ignore
                    self.register_feature(
                        name=registered_feature.name,
                        data=registered_feature.data,
                        feature_level=registered_feature.feature_level,
                        overwrite_previously_registered=True,
                    )
            else:
                setattr(self, key, value)

        self.scaffold_edge_indices = self.compute_edge_indices()
        self.validate()
        logger.info(
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

    def _validate(
        self, expected_num: int, feature_level: FeatureLevel, data_srcs: dict[str, int]
    ):
        if not all_equal(data_srcs.values()):
            msg = (
                f"Expected {expected_num} {feature_level}s but get the following number of "
                f"{feature_level}s from different internal data:\n"
            )

            for name, size in data_srcs.items():
                msg += f"{name}: {size}\n"

            raise RuntimeError(msg)

    def validate(self):
        ### check number of proteins
        num_proteins: dict[str, int] = {
            "num_proteins": self.num_proteins,
            "scaffold_ptr": int(self.scaffold_ptr[-1]),
            "scaffold_sizes": int(self.scaffold_sizes.sum()),
        }

        self._validate(self.num_proteins, "protein", num_proteins)

        ### check number of scaffolds
        scaffold_sizes: dict[str, int] = {
            "num_scaffolds": self.num_scaffolds,
        }
        for field in self.__scaffold_attr__:
            attr_name = field if field.startswith("scaffold") else f"scaffold_{field}"
            attr = getattr(self, attr_name)
            if field == "edge_indices":
                # edge_indices is a list
                size = len(attr)
            else:
                # else it's a tensor
                size = int(attr.shape[0])
            if field == "ptr":
                # ptr has one extra element
                size -= 1
            scaffold_sizes[attr_name] = size

        for name in self._scaffold_registry:
            size = int(self._registered_features[name].data.shape[0])
            scaffold_sizes[f"REGISTRY__{name}"] = size

        self._validate(self.num_scaffolds, "scaffold", scaffold_sizes)

        ### check number of genomes
        genome_sizes: dict[str, int] = {
            "num_genomes": self.num_genomes,
            "genome_is_multiscaffold": int(self.genome_is_multiscaffold.shape[0]),
        }

        for name in self._genome_registry:
            size = int(self._registered_features[name].data.shape[0])
            genome_sizes[f"REGISTRY__{name}"] = size

        self._validate(self.num_genomes, "genome", genome_sizes)

        ### check if any genomes have fewer than 2 proteins
        if (self.num_proteins_per_genome < 2).any():
            raise RuntimeError(
                "Some genomes have fewer than 2 proteins. This is not allowed. These genomes should be excluded."
            )

    def save(self, file: FilePath):
        """Save the dataset to an HDF5 file."""
        with tb.open_file(file, "w") as fp:
            for field in GenomeDataset.__h5_fields__:
                data = getattr(self, field).numpy()
                fp.create_carray("/", field, data, filters=H5_FILE_COMPR_FILTERS)

    def any_multi_scaffold_genomes(self) -> bool:
        return bool(self.genome_is_multiscaffold.any())
