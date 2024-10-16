from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, cast, get_args

import tables as tb
import torch
from torch_geometric.utils import scatter
from tqdm import tqdm

from pst.data.modules import GenomeDataModule
from pst.data.utils import H5_FILE_COMPR_FILTERS, convert_to_scaffold_level_genome_label
from pst.nn.modules import ProteinSetTransformer as PST
from pst.typing import EdgeAttnOutput, GenomeGraphBatch, GraphAttnOutput, PairTensor
from pst.utils.cli.modes import InferenceMode


def model_inference(
    config: InferenceMode,
    node_embeddings: bool = True,
    graph_embeddings: bool = True,
    return_predictions: bool = False,
):
    predictor = Predictor(
        config=config,
        node_embeddings=node_embeddings,
        graph_embeddings=graph_embeddings,
        return_predictions=return_predictions,
    )

    return predictor.predict_loop()


class Predictor:
    OutputType = Literal["node", "graph", "attn"]
    GraphType = Literal["genome", "scaffold", "fragment", "fragmented scaffold"]

    def __init__(
        self,
        config: InferenceMode,
        node_embeddings: bool = True,
        graph_embeddings: bool = True,
        return_predictions: bool = False,
    ):
        self.config = config

        if not node_embeddings and not graph_embeddings:
            raise ValueError(
                "At least one of node_embeddings and graph_embeddings must be True"
            )

        self.node_embeddings = node_embeddings
        self.graph_embeddings = graph_embeddings
        self.return_predictions = return_predictions
        self.allow_genome_fragmenting = config.predict.fragment_oversized_genomes

        # by default, the graph will be "genome" level, assuming that each individual scaffold is
        # a complete genome. If any genomes are multi scaffold, the graph will be "scaffold" level
        # If the dataset was chunked, the graph will be "fragment" level, and if there were multi-scaffold,
        # genomes that get chunked, then the graph will be "fragmented scaffold" level
        self.graph_type: Predictor.GraphType = "genome"

        self.setup()

        self.filters = H5_FILE_COMPR_FILTERS

        # TODO: should just let users provide the file name directly since this is just a single file
        self._file = tb.File(self.config.predict.outdir.joinpath("predictions.h5"), "w")

        self.storage, self.in_memory_storage = self._init_storages()
        self._return_value: dict[str, torch.Tensor] = dict()

    def setup(self):
        self._setup_accelerator()

        ckptfile = self.config.predict.checkpoint
        self._setup_data(ckptfile)
        self._setup_model(ckptfile)

        self.config.predict.outdir.mkdir(parents=True, exist_ok=True)

    def _setup_accelerator(self):
        if self.config.trainer.accelerator == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.config.trainer.accelerator == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device(self.config.trainer.accelerator)

        self.device = device

    def _setup_data(self, ckptfile: Path):
        self.datamodule = GenomeDataModule.from_pretrained(
            checkpoint_path=ckptfile,
            data_file=self.config.data.file,
            command_line_config=self.config.data,  # allow updating batch size and fragment size from cli if set
            shuffle=False,
        )
        self.datamodule.setup("predict")
        if self.datamodule.dataset.any_genomes_have_multiple_scaffolds():
            self.graph_type = "scaffold"

        if "fragment_size" in self.config.data.model_fields_set:
            if self.graph_type == "scaffold":
                self.graph_type = "fragmented scaffold"
            else:
                self.graph_type = "fragment"

    def _setup_model(self, ckptfile: Path):
        self.model = PST.from_pretrained(ckptfile).to(self.device).eval()

        expected_max_size = self.model.positional_embedding.max_size
        actual_max_size = self.datamodule.dataset.max_size
        if actual_max_size > expected_max_size:
            if not self.allow_genome_fragmenting:
                raise RuntimeError(
                    (
                        f"The maximum number of proteins in your dataset is {actual_max_size}"
                        f", but this model was trained with a max of {expected_max_size} "
                        "proteins. If you would like to proceed, set "
                        "`fragment_large_genomes=True` (or at the command line, set --fragment-oversized-genomes)"
                    )
                )

            if self.graph_type == "scaffold":
                self.graph_type = "fragmented scaffold"
            else:
                self.graph_type = "fragment"
            self.datamodule.dataset.fragment(expected_max_size, inplace=True)

    def _create_earray(self, name: OutputType) -> tb.EArray:
        if name == "node":
            loc = "ctx_ptn"
            dim = self.model.encoder.out_dim
            expectedrows = int(self.datamodule.dataset.data.size(0))
        elif name == "graph":
            loc = "data"
            dim = self.model.config.out_dim
            expectedrows = int(self.datamodule.dataset.sizes.numel())
        else:
            loc = "attn"
            dim = self.model.config.num_heads
            expectedrows = int(self.datamodule.dataset.data.size(0))

        earray = self._file.create_earray(
            where=self._file.root,
            name=loc,
            atom=tb.Float32Atom(),
            shape=(0, dim),
            expectedrows=expectedrows,
            filters=self.filters,
        )
        return earray

    def _init_storage(self) -> dict[Predictor.OutputType, tb.EArray]:
        return {
            name: self._create_earray(name) for name in get_args(Predictor.OutputType)
        }

    def _init_in_memory_storage(self) -> dict[Predictor.OutputType, list[torch.Tensor]]:
        return {name: list() for name in get_args(Predictor.OutputType)}

    def _init_storages(self):
        return self._init_storage(), self._init_in_memory_storage()

    def append(self, name: OutputType, data: torch.Tensor):
        data = data.detach().cpu()

        if self.return_predictions:
            self.in_memory_storage[name].append(data)

        self.storage[name].append(data.numpy())

    def get_graph_embeddings(self) -> torch.Tensor:
        if self.return_predictions:
            return torch.cat(self.in_memory_storage["graph"])

        # if not returning predictions, then the data is only in the file
        if not self._file.isopen:
            with tb.open_file(self._file.filename, "r") as fp:
                data = fp.root.data[:]
        else:
            data = self._file.root.data[:]

        return torch.from_numpy(data)

    def _reduce_fragments_to_scaffolds(self, reduce: str = "mean") -> PairTensor:
        # self.graph_type is either "fragment" or "fragmented scaffold"
        # shape: [N chunks, D]
        fragment_embeddings = self.get_graph_embeddings()

        # so we need to reduce fragment level embeddings to scaffold level
        # scaffolds could either be the entire complete genome or a single scaffold in a multi-scaffold genome

        # shape: [N scaffolds, D]
        scaffold_embeddings = scatter(
            fragment_embeddings,
            self.datamodule.dataset.scaffold_label,
            reduce=reduce,
        )

        return fragment_embeddings, scaffold_embeddings

    def reduce_fragmented_genomes(self, reduce: str = "mean") -> PairTensor:
        # self.graph_type = "fragment", ie the graph embeddings are for fragments of scaffolds
        # that represent entire genomes, not individual scaffolds in a multi-scaffold genome

        return self._reduce_fragments_to_scaffolds(reduce=reduce)

    def reduce_fragmented_multiscaffold_genomes(
        self, reduce: str = "mean"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # self.graph_type = "fragmented scaffold", ie the graph embeddings are for fragments of scaffolds
        # that represent individual scaffolds in a multi-scaffold genomes

        fragment_embeddings, scaffold_embeddings = self._reduce_fragments_to_scaffolds(
            reduce=reduce
        )

        # this is needed since the genome_label labels each individual FRAGMENT
        # so we need to convert it to labeling scaffolds
        # ie shape: [N fragments] -> [N scaffolds]
        scaffold_level_genome_label = convert_to_scaffold_level_genome_label(
            self.datamodule.dataset.genome_label,
            self.datamodule.dataset.scaffold_label,
        )

        # shape: [N genomes, D]
        genome_embeddings = scatter(
            scaffold_embeddings, scaffold_level_genome_label, reduce=reduce
        )

        return fragment_embeddings, scaffold_embeddings, genome_embeddings

    def reduce_scaffolds(self, reduce: str = "mean") -> PairTensor:
        # self.graph_type is "scaffold", ie the graph embeddings are at the scaffold level
        # for multi-scaffold genomes

        scaffold_embeddings = self.get_graph_embeddings()
        genome_embeddings = scatter(
            scaffold_embeddings, self.datamodule.dataset.genome_label, reduce=reduce
        )

        return scaffold_embeddings, genome_embeddings

    def _add_reduced_embeddings_to_storages(
        self,
        name: str,
        embedding: torch.Tensor,
    ):
        self._file.create_carray(
            where=self._file.root,
            name=name,
            obj=embedding.numpy(),
            filters=self.filters,
        )

        # self.in_memory_storage[name] = embedding  # type: ignore
        self._return_value[name] = embedding

    def _postprocess_in_memory_storage(self):
        # the values in self.in_memory_storage are always a list of tensors
        # we need to concatenate them to a single tensor if returning predictions
        # to a python caller

        node_embeddings = torch.cat(self.in_memory_storage.pop("node"))
        self._return_value["node"] = node_embeddings

        attn = torch.cat(self.in_memory_storage["attn"])
        self._return_value["attn"] = attn

        # this is called after reducing the embeddings to the desired level
        # so self.in_memory_storage will have the appropriate keys for
        # the embeddings at each level.
        # NOTE: all the default keys should be missing from self.in_memory_storage

        graph_embeddings = self.in_memory_storage.pop("graph")
        if self.graph_type == "genome":
            # self.in_memory_storage["graph"] is already at the genome level
            genome_embeddings = torch.cat(graph_embeddings)
            self._return_value["genome"] = genome_embeddings

        # for all other graph types, the embeddings are at a higher level
        # and have already been reduced to the desired level and added to the
        # return value storage

    def _rename_graph_embedding_node_in_h5_file(self):
        # by default the graph embeddings are stored in the "data" node in the h5 file
        # but not only is this redundant since the other data for different reduced
        # levels are stored in separate nodes, with one of them being identical to the
        # "data" node, but it's also confusing to know what is in the "data" node
        # graph_type = "scaffold" -> data = scaffold
        # graph_type = "fragment" or "fragmented scaffold" -> data = fragment
        # graph_type = "genome" -> data = genome
        # however, if graph_type != "genome", then the redundant field is already there
        # so just remove the data node
        # but for graph_type == "genome", we need to rename the data node to genome

        if self.graph_type == "genome":
            # data = genome
            self._file.rename_node(where=self._file.root, name="data", newname="genome")
        else:
            self._file.remove_node(where=self._file.root, name="data")

    @torch.no_grad()
    def predict_loop(self) -> Optional[dict[str, torch.Tensor]]:
        # implemented a barebones loop instead of using lightning to enable saving
        # both the ptn embeddings and the graph embeddings separately
        dataloader = self.datamodule.predict_dataloader()

        batch: GenomeGraphBatch
        for batch in tqdm(dataloader, file=sys.stdout):
            batch = batch.to(self.device)  # type: ignore

            x_cat, _, _ = self.model.internal_embeddings(batch)

            node_output: EdgeAttnOutput = self.model.encoder(
                x_cat, batch.edge_index, batch.batch
            )

            if self.node_embeddings:
                self.append(name="node", data=node_output.out)

            graph_output: GraphAttnOutput = self.model.decoder(
                x=node_output.out,
                ptr=batch.ptr,
                batch=batch.batch,
                return_attention_weights=True,
            )

            if self.graph_embeddings:
                self.append(name="graph", data=graph_output.out)
                attn = cast(torch.Tensor, graph_output.attn)
                self.append(name="attn", data=attn)

        reduced_embeddings: dict[str, torch.Tensor] = dict()
        # if self.graph_type == "genome" ### don't need to do anything extra
        # since the graph embeddings are already genome level
        if self.graph_type == "scaffold":
            # the graph embeddings are at the scaffold level (ie multi-scaffold genomes)
            # need to reduce scaffold level embeddings to genome level
            scaffold_embeddings, genome_embeddings = self.reduce_scaffolds()
            reduced_embeddings["genome"] = genome_embeddings
            reduced_embeddings["scaffold"] = scaffold_embeddings
        elif self.graph_type == "fragment":
            # the graph embeddings are at the fragment level
            # need to reduce fragment level embeddings to scaffold level
            # here, the scaffold level embeddings are the same as genome level
            fragmented_embeddings, genome_embeddings = self.reduce_fragmented_genomes()
            reduced_embeddings["fragment"] = fragmented_embeddings
            reduced_embeddings["genome"] = genome_embeddings
        elif self.graph_type == "fragmented scaffold":
            # the graph embeddings are at the fragmented scaffold level
            # need to reduce fragmented scaffold level embeddings to scaffold level
            # then reduce scaffold level embeddings to genome level
            fragmented_embeddings, scaffold_embeddings, genome_embeddings = (
                self.reduce_fragmented_multiscaffold_genomes()
            )
            reduced_embeddings["fragment"] = fragmented_embeddings
            reduced_embeddings["scaffold"] = scaffold_embeddings
            reduced_embeddings["genome"] = genome_embeddings

        for name, embedding in reduced_embeddings.items():
            self._add_reduced_embeddings_to_storages(name, embedding)

        self._rename_graph_embedding_node_in_h5_file()

        self.close()

        if self.return_predictions:
            self._postprocess_in_memory_storage()
            self.in_memory_storage.clear()
            return self._return_value
        return None

    def close(self):
        self._file.close()
