import logging
from collections.abc import Iterable
from functools import cached_property
from typing import Any, Sequence, Union, overload

import tables as tb
import torch
from more_itertools import all_equal
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from typing_extensions import deprecated

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
    _merge_indices,
    _ScaffoldFeatureFragmentedData,
    graph_sizes_to_index_pointer,
)
from pst.typing import EdgeIndexStrategy, FilePath, GenomeGraphBatch, OptTensor

_SENTINEL_FRAGMENT_SIZE = -1
logger = logging.getLogger(__name__)

_IndexT = Union[int, slice, Iterable[int]]


class _BaseGenomeDataset(Dataset[GenomeGraphBatch]):
    # The names of protein/scaffold/genome attributes refer to what they are labeling and thus what
    # the expected shape of each tensor should be. For example, "protein_data" should be of shape:
    # [num proteins, D]. The __X_attr__ class attributes define the attributes at each level.

    __minimum_h5_fields__ = {"data", "sizes", "strand"}
    __expected_h5_fields__ = __minimum_h5_fields__ | {
        "ptr",
        "scaffold_label",
        "genome_label",
    }

    # the names of the object attributes at each level will be prefixed with the level name
    # if you want the protein embeddings, you can refer to `dataset.protein_data`
    __protein_attr__ = {"x", "data", "strand"}
    __scaffold_attr__ = {
        "ptr",
        "sizes",
        # "class_id",
        "scaffold_label",
        "genome_label",
        # "edge_indices", # not part of the Lazy class
        "part_of_multiscaffold",
    }
    __genome_attr__ = {"is_multiscaffold"}

    # ------------- Protein/Scaffold/Genome level attributes -------------
    # shapes should all refer to number of proteins
    # protein_data: Tensor
    protein_data: tb.Array
    protein_strand: Tensor

    # shapes should all refer to number of scaffolds
    scaffold_ptr: Tensor
    scaffold_sizes: Tensor
    # scaffold_edge_indices: list[Tensor]
    scaffold_label: Tensor
    scaffold_genome_label: Tensor
    scaffold_part_of_multiscaffold: Tensor

    # shapes should all refer to number of genomes
    genome_is_multiscaffold: Tensor
    # --------------------------------------------------------------------

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
        fragment_size: int = _SENTINEL_FRAGMENT_SIZE,
        lazy: bool = True,
    ) -> None:
        super().__init__()
        self.lazy = lazy

        self._setup_registry()
        self._read_h5_file(file)
        self._compute_missing_attribute_tensors()
        self._convert_strand_tensor()

        if self.any_multi_scaffold_genomes():
            logger.info("Multi-scaffold genomes detected.")

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

            # defer validation for eager dataset subclasses that fully load data
            self.fragment(fragment_size, inplace=True, validate=False)
            self._fragmented = True
        else:
            # .fragment will compute edge indices and call .validate
            self._fragmented = False

        if self.lazy:
            self.validate()

    def _setup_registry(self):
        # just keep track of names
        self._genome_registry: set[str] = set()
        self._scaffold_registry: set[str] = set()
        self._protein_registry: set[str] = set()

        self._registered_features: dict[str, RegisteredFeature] = dict()

    def _read_h5_file(self, file: FilePath):
        # for lazy loading, self.protein_data points to the h5 file Array
        features_to_register: list[dict[str, Any]] = []
        protein_data_node_path = None
        self._file = tb.open_file(file)  # type: ignore

        for node in self._file.root:
            name: str = node._v_name  # type: ignore

            # allow subgroups in h5 file, all data fields beneath them will be registered
            if isinstance(node, tb.Group):
                if name not in _BaseGenomeDataset.__feature_levels__:
                    raise ValueError(
                        f"Unknown feature level {name}. Subgroups must be one of {_BaseGenomeDataset.__feature_levels__}"
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
                if name in _BaseGenomeDataset.__expected_h5_fields__:
                    # directly set these as attributes
                    if name in _BaseGenomeDataset.__protein_attr__:
                        prefix = "protein"
                    elif name in _BaseGenomeDataset.__scaffold_attr__:
                        prefix = "scaffold"
                    # there are no expected genome fields
                    else:
                        assert False, "UNREACHABLE"

                    obj_attr_name = name if name == "scaffold_label" else f"{prefix}_{name}"

                    if obj_attr_name != "protein_data":
                        setattr(self, obj_attr_name, torch.from_numpy(node[:]))  # type: ignore
                    else:
                        protein_data_node_path = node._v_pathname
                else:
                    # then other top level fields will become a registered feature
                    # so self._setup_registry() should be called before this method
                    # these are REQUIRED to label the name with "protein_", "scaffold_", or "genome_"
                    feature_level = name.split("_", 1)[0]
                    if feature_level not in _BaseGenomeDataset.__feature_levels__:
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
                            "data": torch.from_numpy(node[:]),  # type: ignore
                            "feature_level": feature_level,
                        }
                    )

        if protein_data_node_path is None:
            raise ValueError(
                "Could not find protein embedding data in the h5 file. Please check the file format."
            )

        if protein_data_node_path.count("/") > 1:
            protein_data_where, protein_data_name = protein_data_node_path.rsplit("/", 1)
        else:
            protein_data_where = "/"
            protein_data_name = protein_data_node_path[1:]  # remove leading "/"

        self.protein_data = self._file.get_node(protein_data_where, protein_data_name)  # type: ignore

        # now need to check that the minimum required fields are set
        missing: set[str] = set()
        for name in _BaseGenomeDataset.__minimum_h5_fields__:
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

        if not hasattr(self, "scaffold_ptr"):
            # just compute it from scaffold sizes
            self.scaffold_ptr = graph_sizes_to_index_pointer(self.scaffold_sizes)

        for feature in features_to_register:
            self.register_feature(**feature)

    def _compute_missing_attribute_tensors(self):
        # check that all attributes exist
        if not hasattr(self, "scaffold_genome_label"):
            # assume each scaffold is a separate genome
            self.scaffold_genome_label = torch.arange(len(self.scaffold_sizes), dtype=torch.long)

        if not hasattr(self, "scaffold_label"):
            self.scaffold_label = torch.arange(len(self.scaffold_sizes), dtype=torch.long)

        # shape: [Num genomes]
        self.genome_is_multiscaffold = torch.bincount(self.scaffold_genome_label) > 1

        # shape: [Num scaffolds]
        self.scaffold_part_of_multiscaffold = self.genome_is_multiscaffold[
            self.scaffold_genome_label
        ]

    def _convert_strand_tensor(self):
        # convert strand array from [-1, 1] -> [0, 1]
        # this will be used as an idx in a lut embedding
        self.protein_strand[self.protein_strand == -1] = 0

    def _standardize_indices(self, idx: int | slice | Iterable[int]) -> Iterable[int]:
        # this will convert all indices to a list of integers
        if isinstance(idx, int):
            return [idx]

        if isinstance(idx, slice):
            return range(*idx.indices(len(self)))

        if isinstance(idx, torch.Tensor):
            return idx.tolist()

        if isinstance(idx, Iterable):
            return list(idx)

        raise ValueError(f"Invalid index type {type(idx)}")

    def _compute_edge_indices(self, idx: _IndexT | None = None) -> list[Tensor]:
        if idx is None:
            # compute all edge indices
            indices = range(len(self))
        else:
            indices = self._standardize_indices(idx)

        # needed to compute edge indices for new fragmented datasets
        edge_indices: list[Tensor] = list()

        for index in indices:
            num_nodes = int(self.scaffold_sizes[index])
            is_multi_scaffold = bool(self.scaffold_part_of_multiscaffold[index])

            try:
                edge_index = self.edge_create_fn(num_nodes=num_nodes)
            except ValueError as e:
                # this only occurs when the scaffold has 1 protein
                if is_multi_scaffold or self._fragmented:
                    edge_index = torch.tensor([[0, 0]]).t().contiguous()
                else:
                    raise RuntimeError(
                        "Failed to create edge index, since a scaffold only contains 1 protein. "
                        "This is only allowed for scaffold that are part of multi-scaffold "
                        "genomes or if the dataset was artificially fragmented."
                    ) from e

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
    def num_proteins_per_genome(self) -> Tensor:
        return scatter(self.scaffold_sizes, self.scaffold_genome_label, reduce="sum")

    def _get_scaffold_data(self, idx: Iterable[int]) -> dict[str, list[torch.Tensor]]:
        # this will return all scaffold-level data for the given index
        # idx should be Iterable[int] now

        scaffold_data: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                # "edge_index",
                "num_proteins",
                "scaffold_label",
                "genome_label",
                "pos",
                "scaffold_idx",
            )
        }

        for name in self._scaffold_registry:
            scaffold_data[name] = []

        scaffold_edge_indices = self._compute_edge_indices(idx)

        scaffold_data["edge_index"] = scaffold_edge_indices

        for i in idx:
            num_proteins = self.scaffold_sizes[i]
            scaffold_data["num_proteins"].append(num_proteins)
            pos = (
                torch.arange(int(num_proteins)).unsqueeze(-1).to(self.scaffold_ptr.device)
            )  # dont think device matters here since the batch will be moved later
            scaffold_data["pos"].append(pos)
            scaffold_data["scaffold_label"].append(self.scaffold_label[i])
            scaffold_data["genome_label"].append(self.scaffold_genome_label[i])

            for name in self._scaffold_registry:
                data = self._registered_features[name].data
                scaffold_data[name].append(data[i])

            scaffold_data["scaffold_idx"].append(torch.tensor(i))

        # could concat some of these, but this is all handled better by the GenomeGraph
        # batching system
        return scaffold_data

    def _get_protein_data(self, idx: Iterable[int]) -> dict[str, list[torch.Tensor]]:
        # these slices refer to scaffolds, not proteins, so now we need to convert
        # to protein indices
        scaffold_slices = _merge_indices(idx)
        protein_slices: list[slice] = []
        for scaffold_slice in scaffold_slices:
            start = self.scaffold_ptr[scaffold_slice.start]
            stop = self.scaffold_ptr[scaffold_slice.stop]
            protein_slices.append(slice(start, stop))

        batched_protein_data: dict[str, list[torch.Tensor]] = {name: [] for name in ("x", "strand")}

        for name in self._protein_registry:
            batched_protein_data[name] = []

        for slc in protein_slices:
            # if lazy x will be a numpy FloatArray
            # else it will be a torch tensor
            x = self.protein_data[slc]
            if self.lazy:  # x will be loaded from file as a numpy FloatArray
                x = torch.from_numpy(x)

            strand = self.protein_strand[slc]
            batched_protein_data["x"].append(x)
            batched_protein_data["strand"].append(strand)

            for name in self._protein_registry:
                data = self._registered_features[name].data
                batched_protein_data[name].append(data[slc])

        return batched_protein_data

    def _unbatch_protein_data(
        self, batched_protein_data: dict[str, list[Tensor]], num_proteins_per_scaffold: Tensor
    ) -> dict[str, list[Tensor]]:
        """NOTE: clears the input `batched_protein_data` dict to avoid storing unnecessary data"""
        # we need to unbatch the data here so that each pos in the protein_data list points to an
        # individual scaffold.
        # this is bc we need to create individual GenomeGraph objects

        stacked_protein_data = {
            key: torch.concat(tensor_list) for key, tensor_list in batched_protein_data.items()
        }

        batched_protein_data.clear()

        protein_data: dict[str, list[torch.Tensor]] = {
            key: [] for key in stacked_protein_data.keys()
        }

        n_scaffolds = num_proteins_per_scaffold.numel()
        scaffold_ptr = graph_sizes_to_index_pointer(num_proteins_per_scaffold)

        for scaffold_idx in range(n_scaffolds):
            start = scaffold_ptr[scaffold_idx]
            stop = scaffold_ptr[scaffold_idx + 1]

            for key, tensor in stacked_protein_data.items():
                protein_data[key].append(tensor[start:stop])

        return protein_data

    @overload
    def __getitem__(self, idx: int) -> GenomeGraph: ...

    @overload
    def __getitem__(self, idx: slice) -> GenomeGraphBatch: ...

    @overload
    def __getitem__(self, idx: Iterable[int]) -> GenomeGraphBatch: ...

    def __getitem__(self, idx: _IndexT) -> GenomeGraph | GenomeGraphBatch:
        # this is scaffold-level access, for genome-level access use .get_genome

        idx = self._standardize_indices(idx)
        if any(i >= len(self) for i in idx):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        scaffold_data = self._get_scaffold_data(idx)
        batched_protein_data = self._get_protein_data(idx)

        num_proteins_per_scaffold = torch.tensor(scaffold_data["num_proteins"])
        protein_data = self._unbatch_protein_data(batched_protein_data, num_proteins_per_scaffold)

        graph_data = scaffold_data | protein_data  # works py 3.9+

        # validate the data
        num_scaffolds = num_proteins_per_scaffold.numel()
        for key, tensor_list in graph_data.items():
            if len(tensor_list) != num_scaffolds:
                raise ValueError(
                    f"Expected {num_scaffolds} tensors for key {key}, but got {len(tensor_list)}"
                )

        genome_graphs: list[GenomeGraph] = []
        for scaffold_idx in range(num_scaffolds):
            # create a new genome graph object for each scaffold
            data = {key: tensor_list[scaffold_idx] for key, tensor_list in graph_data.items()}
            genome_graph = GenomeGraph(**data)  # type: ignore
            genome_graphs.append(genome_graph)

        if len(genome_graphs) == 1:
            # if only one genome graph, return it directly
            return genome_graphs[0]

        return self.collate(genome_graphs)

    def get_genome(self, genome_id: int) -> GenomeGraphBatch:
        # always return a batch obj even if only one genome
        # the batch obj has all the methods of the single graph obj

        genome_idx = torch.where(self.scaffold_genome_label == genome_id)[0]
        genome_idx = genome_idx.tolist()
        return self[genome_idx]

    def _register_feature(self, name: str, data: Tensor, feature_level: FeatureLevel):
        # feature_level is one of {protein, scaffold, genome}

        name_registry: set[str] = getattr(self, f"_{feature_level}_registry")
        name_registry.add(name)

        feature = RegisteredFeature(name=name, data=data, feature_level=feature_level)
        self._registered_features[name] = feature

    def register_feature(
        self,
        name: str,
        data: Tensor,
        *,
        feature_level: FeatureLevel,
        overwrite_previously_registered: bool = False,
    ):
        if name in (self.__protein_attr__ | self.__scaffold_attr__ | self.__genome_attr__):
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

    def get_registered_feature(self, name: str) -> Tensor:
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

        # TODO: there is a bug when collate genome batches instead of scaffold batches
        # since .get_genome uses this fn all genome labels become 0...
        _, rel_scaffold_label = databatch.scaffold_label.unique(sorted=False, return_inverse=True)
        _, rel_genome_label = databatch.genome_label.unique(sorted=False, return_inverse=True)

        # scaffold_label and genome_label are not really relevant at the minibatch level
        # they are more useful at the dataset level where we can artificially fragment
        # genomes. In these scenarios, we can keep track of which fragments belong to which
        # scaffolds, and which scaffolds belong to which genomes (if we have metagenomes).
        # however, they're still provided at the minibatch level for convenience

        # shape: [N scaffolds] each
        # keep these separate since the global
        # setattr(databatch, "rel_scaffold_label", rel_scaffold_label)
        # setattr(databatch, "rel_genome_label", rel_genome_label)

        # shape: [N scaffolds]
        databatch.scaffold_label = rel_scaffold_label

        # TODO: update naming?
        # shape: [N scaffolds]
        databatch.genome_label = rel_genome_label

        return databatch

    def collate_indices(self, idx_batch: list[Tensor]) -> GenomeGraphBatch:
        batch = [self[int(idx)] for idx in idx_batch]
        return self.collate(batch)

    def _resize_scaffold_features(
        self, chunk_scaffold_labels: Tensor
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

    def _update_from_fragmented_data(self, fragmented_data: _FragmentedData, validate: bool = True):
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

        # need to set _fragmented to True for edge index creation to not raise errors for
        # single protein fragments. this is preferred over merging single protein fragment with
        # a previous fragment since the fragment size may be a hard maximum, like when adjusting
        # due to model positional embedding LUT
        # the other alternative would be modifying the user input fragment size to be the next
        # smallest integer that does not produce single protein fragments
        # -> note that this issue only affects genomes that were not originally multiscaffold
        self._fragmented = True

        if validate:
            self.validate()

        logger.info(
            f"Fragmented dataset into {len(self)} fragments with <= {self.max_size} proteins."
        )

    def fragment(self, max_size: int, inplace: bool = True, validate: bool = True):
        fragmented_data = self._fragment(max_size)

        if inplace:
            self._update_from_fragmented_data(fragmented_data, validate)
            return self

        from copy import copy, deepcopy

        # this makes a shallow copy, so the underlying tensors are actually shared
        # but we can still reassign the tensors. Most signficant reason to do this
        # is to not have to copy the protein embeddings
        new_dataset = copy(self)
        # however, we need to make a deep copy of the registered features
        # so that the new and old datasets don't share the same dict
        new_dataset._registered_features = deepcopy(self._registered_features)
        new_dataset._update_from_fragmented_data(fragmented_data, validate)

        return new_dataset

    def _validate(self, expected_num: int, feature_level: FeatureLevel, data_srcs: dict[str, int]):
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
        with tb.open_file(file, "w", filters=H5_FILE_COMPR_FILTERS) as fp:  # type: ignore
            for field in _BaseGenomeDataset.__expected_h5_fields__:
                data = getattr(self, field).numpy()
                fp.create_carray("/", field, data)

            groups: dict[str, tb.Group] = dict()
            for feature in self._registered_features.values():
                group_name = feature.feature_level
                group = groups.get(
                    group_name,
                    fp.create_group(fp.root, group_name),
                )

                fp.create_carray(group, feature.name, feature.data.numpy())

    def any_multi_scaffold_genomes(self) -> bool:
        return bool(self.genome_is_multiscaffold.any())

    def __getattr__(self, name: str):
        try:
            return self.get_registered_feature(name)
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}', "
                "and this is not a registered feature"
            ) from e

    @classmethod
    def _init_arg_names(cls) -> set[str]:
        import inspect

        return set(inspect.signature(cls).parameters.keys())

    ### backwards compatibility --- all marked as deprecated

    @property
    @deprecated(
        "Deprecated since v1.3: Use .protein_data instead for clarity",
        category=DeprecationWarning,
    )
    def data(self) -> Tensor:
        """Protein embeddings, shape: [num proteins, D]"""
        return self.protein_data

    @property
    @deprecated(
        "Deprecated since v1.3: Use .scaffold_ptr instead for clarity",
        category=DeprecationWarning,
    )
    def ptr(self) -> Tensor:
        """CSR Index pointer to the start of each set of protein embeddings in
        self.protein_data for each scaffold, shape: [num scaffolds + 1]"""
        return self.scaffold_ptr

    @property
    @deprecated(
        "Deprecated since v1.3: Use .scaffold_sizes instead for clarity",
        category=DeprecationWarning,
    )
    def sizes(self) -> Tensor:
        """Number of proteins in each scaffold, shape: [num scaffolds]"""
        return self.scaffold_sizes

    @property
    @deprecated(
        (
            "Deprecated since v1.3: This is no longer a required field for a GenomeDataset. It "
            "is instead a registered feature. Use .get_registered_feature('class_id') instead "
            "(or whatever name you have registered it as)."
        ),
        category=DeprecationWarning,
    )
    def class_id(self) -> OptTensor:
        """Class ID for each scaffold if it exists, shape: [num scaffolds]"""
        try:
            return self.get_registered_feature("class_id")
        except KeyError:
            return None

    @property
    @deprecated(
        (
            "Deprecated since v1.3: This is no longer a required field for a GenomeDataset. It "
            "is instead a registered feature. Use .get_registered_feature('weights') instead "
            "(or whatever name you have registered it as)."
        ),
        category=DeprecationWarning,
    )
    def weight(self) -> OptTensor:
        """Class weight for each scaffold if it exists, shape: [num scaffolds]"""
        try:
            return self.get_registered_feature("weight")
        except KeyError:
            return None

    @property
    @deprecated(
        "Deprecated since v1.3: Use .protein_strand instead for clarity",
        category=DeprecationWarning,
    )
    def strand(self) -> Tensor:
        """Strand of each protein, shape: [num proteins]"""
        return self.protein_strand

    @property
    @deprecated(
        "Deprecated since v1.3: Use .scaffold_genome_label instead for clarity",
        category=DeprecationWarning,
    )
    def genome_label(self) -> Tensor:
        """Genome label for each scaffold, shape: [num scaffolds]"""
        return self.scaffold_genome_label


class LazyGenomeDataset(_BaseGenomeDataset):
    def __init__(
        self,
        file: FilePath,
        edge_strategy: EdgeIndexStrategy = _DEFAULT_EDGE_STRATEGY,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
        fragment_size: int = _SENTINEL_FRAGMENT_SIZE,
    ) -> None:
        super().__init__(
            file=file,
            edge_strategy=edge_strategy,
            chunk_size=chunk_size,
            threshold=threshold,
            fragment_size=fragment_size,
            lazy=True,
        )

    def _convert_strand_tensor(self):
        # [-1, 1] -> [0, 1]
        super()._convert_strand_tensor()

        # lazy loading suggests that memory is an issue, so we can save 8x memory by
        # storing the strand tensor as a bool compared to int64

        self.protein_strand = self.protein_strand.bool()

        # then we just need to convert it back when needed for the strand embedding lut

    def _get_protein_data(self, idx: Iterable[int]) -> dict[str, list[Tensor]]:
        batched_protein_data = super()._get_protein_data(idx)

        # convert bool strand back to int64 for indexing
        batched_protein_data["strand"] = [
            strand.long() for strand in batched_protein_data["strand"]
        ]

        return batched_protein_data


class GenomeDataset(_BaseGenomeDataset):
    """Genome dataset that fully loads all protein embeddings and edge indices into memory.

    If memory is not an issue, this is the preferred way to use the dataset as it is faster,
    especially during training. For inference, where each genome only needs to be loaded once,
    eager vs lazy loading should not make a significant difference.

    If memory is an issue, prefer `LazyGenomeDataset` instead.
    """

    protein_data: Tensor

    scaffold_edge_indices: list[Tensor]

    # add edge_indices
    __scaffold_attr__ = _BaseGenomeDataset.__scaffold_attr__ | {
        "edge_indices",
    }

    def __init__(
        self,
        file: FilePath,
        edge_strategy: EdgeIndexStrategy = _DEFAULT_EDGE_STRATEGY,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        threshold: int = _SENTINEL_THRESHOLD,
        fragment_size: int = _SENTINEL_FRAGMENT_SIZE,
    ) -> None:
        super().__init__(
            file=file,
            edge_strategy=edge_strategy,
            chunk_size=chunk_size,
            threshold=threshold,
            fragment_size=fragment_size,
            lazy=False,
        )

        self.protein_data = torch.from_numpy(self.protein_data[:])
        # no longer need the file handle open
        self._file.close()

        # compute and store edge indices for all scaffolds
        self.scaffold_edge_indices = self._compute_edge_indices()

        # we can now validate the dataset since edge indices should exist
        self.validate()

    @property
    @deprecated(
        "Deprecated since v1.3: Use .scaffold_edge_indices instead for clarity",
        category=DeprecationWarning,
    )
    def edge_indices(self) -> list[torch.Tensor]:
        """Edge indices list, list shape: [num scaffolds] where each edge index tensor has
        shape [2, E] where E is the number of edges in the scaffold based on the number of proteins.
        """
        return self.scaffold_edge_indices


# only used for type hints
class SubsetGenomeDataset(Subset[GenomeGraphBatch]):
    dataset: LazyGenomeDataset
    indices: Sequence[int]

    def __init__(self, dataset: GenomeDataset, indices: Sequence[int]):
        super().__init__(dataset, indices)


class GenomeSubset(SubsetGenomeDataset):
    def __getitem__(self, idx):
        # self.indices points to genome indices, which doesn't work with GenomeDataset.__getitem__

        return self.dataset.get_genome(self.indices[idx])
