import logging
import sys
from pathlib import Path
from typing import Literal, Optional, get_args

import tables as tb
import torch
from torch_geometric.utils import scatter, segment
from tqdm import tqdm

from pst.data.dataset import _SENTINEL_FRAGMENT_SIZE
from pst.data.modules import GenomeDataModule
from pst.data.utils import (
    H5_FILE_COMPR_FILTERS,
    convert_to_scaffold_level_genome_label,
)
from pst.nn.base import BaseModelTypes, BaseProteinSetTransformer
from pst.nn.modules import ProteinSetTransformer
from pst.typing import EdgeAttnOutput, GenomeGraphBatch, GraphAttnOutput, PairTensor
from pst.utils.cli.predict import PredictArgs
from pst.utils.cli.trainer import AcceleratorOpts

logger = logging.getLogger(__name__)


class PredictMode:
    def predict(
        self,
        file: Path,
        predict: PredictArgs,
        model_type: BaseModelTypes = ProteinSetTransformer,
        accelerator: AcceleratorOpts = AcceleratorOpts.auto,
        devices: int = 1,
        batch_size: Optional[int] = None,
        fragment_size: Optional[int] = None,
        lazy: bool = True,
        return_protein_embeddings: bool = True,
        return_genome_embeddings: bool = True,
    ) -> Optional[dict[str, torch.Tensor]]:
        """PST predict mode for predicting with a pretrained Protein Set Transformer

        Args:
            file (Path): path to the graph-formatted .h5 data file
            predict (PredictArgs): predict configuration
            model_type (BaseModelTypes, optional): PST model type to use for prediction
            accelerator (AcceleratorOpts, optional): accelerator to use
            devices (int, optional): number of devices to use for prediction depending on the
                accelerator. For GPUs, this is the number of GPU devices to use. For CPU-only
                servers, this is the number of CPU cores to use.
            batch_size (Optional[int], optional): batch size to use for prediction. Defaults to
                batch size in checkpoint file.
            fragment_size (Optional[int], optional): fragment size to use for prediction.
                Defaults to fragment size in checkpoint file.
            lazy (bool, optional): whether to use lazy loading for the dataset.
                If True, the dataset will be loaded lazily, which can save memory
                but may be marginally slower. If False, the dataset will be fully loaded into memory
                before prediction. --lazy true is HIGHLY recommended for large microbial datasets or if memory is an issue.
                Defaults to True.
            return_protein_embeddings (bool, optional): whether to return protein embeddings.
                Defaults to True.
            return_genome_embeddings (bool, optional): whether to return genome embeddings.
                Defaults to True.
        """
        model_inference(
            model_type=model_type,
            file=file,
            predict=predict,
            accelerator=accelerator,
            devices=devices,
            batch_size=batch_size,
            fragment_size=fragment_size,
            lazy=lazy,
            node_embeddings=return_protein_embeddings,
            graph_embeddings=return_genome_embeddings,
            return_predictions=False,  # don't store in mem if using CLI
        )


def model_inference(
    model_type: BaseModelTypes,
    file: Path,
    predict: PredictArgs,
    accelerator: AcceleratorOpts = AcceleratorOpts.auto,
    devices: int = 1,
    batch_size: Optional[int] = None,
    fragment_size: Optional[int] = None,
    lazy: bool = False,
    node_embeddings: bool = True,
    graph_embeddings: bool = True,
    return_predictions: bool = False,
):
    logger.info("Starting model inference.")
    predictor = Predictor(
        model_type=model_type,
        file=file,
        predict=predict,
        accelerator=accelerator,
        devices=devices,
        batch_size=batch_size,
        fragment_size=fragment_size,
        lazy=lazy,
        node_embeddings=node_embeddings,
        graph_embeddings=graph_embeddings,
        return_predictions=return_predictions,
    )

    dataset = predictor.datamodule.dataset
    n_genomes = dataset.num_genomes

    if "fragment" not in predictor.graph_type:
        n_scaffolds = dataset.num_scaffolds
        n_chunks = 0
    else:
        n_chunks = dataset.num_scaffolds
        n_scaffolds = dataset.scaffold_label.amax() + 1

    msg = f"Starting prediction loop on {n_genomes} genomes composed of {n_scaffolds} scaffolds"

    if n_chunks > 0:
        msg += f" that were broken into {n_chunks} chunks."

    logger.info(msg)

    if lazy:
        logger.info(
            "Lazy loading is enabled. This may slow down the prediction process, but will save memory."
        )

    results = predictor.predict_loop()
    return results


# TODO: This is only an embedding predictor, so we don't really need to consider if using a GenomeLoader?
# if we did, then the expected sizes would be wrong
class Predictor:
    OutputType = Literal["node", "graph", "attn"]
    GraphType = Literal["genome", "scaffold", "fragment", "fragmented scaffold"]

    def __init__(
        self,
        model_type: BaseModelTypes,
        file: Path,
        predict: PredictArgs,
        accelerator: AcceleratorOpts = AcceleratorOpts.auto,
        devices: int = 1,
        batch_size: Optional[int] = None,
        fragment_size: Optional[int] = None,
        lazy: bool = False,
        node_embeddings: bool = True,
        graph_embeddings: bool = True,
        return_predictions: bool = False,
    ):
        self.model_type = model_type
        self.input_file = file
        self.accelerator = accelerator
        self.predict_cfg = predict
        self.batch_size = batch_size
        self.fragment_size = fragment_size
        self.lazy = lazy

        if not node_embeddings and not graph_embeddings:
            raise ValueError(
                "At least one of node_embeddings and graph_embeddings must be True"
            )

        self.node_embeddings = node_embeddings
        self.graph_embeddings = graph_embeddings

        if not self.node_embeddings and not self.graph_embeddings:
            raise ValueError(
                "At least one of node_embeddings and graph_embeddings must be True. Otherwise, no embeddings will be returned or stored."
            )

        self.return_predictions = return_predictions
        self.allow_genome_fragmenting = self.predict_cfg.fragment_oversized_genomes

        # by default, the graph will be "genome" level, assuming that each individual scaffold is
        # a complete genome. If any genomes are multi scaffold, the graph will be "scaffold" level
        # If the dataset was chunked, the graph will be "fragment" level, and if there were multi-scaffold,
        # genomes that get chunked, then the graph will be "fragmented scaffold" level
        self.graph_type: Predictor.GraphType = "genome"

        self.setup()

        # don't accidentally use CPU threads when GPUs available
        # there are some components to batching that will use CPU threads...so just in case
        torch.set_num_threads(devices if self.device.type == "cpu" else 1)

        self.filters = H5_FILE_COMPR_FILTERS
        self._file = tb.File(self.predict_cfg.output, "w", filters=self.filters)  # type: ignore

        self.storage, self.in_memory_storage = self._init_storages()
        self._return_value: dict[str, torch.Tensor] = dict()

    def setup(self):
        self._setup_accelerator()

        ckptfile = self.predict_cfg.checkpoint
        self._setup_data(ckptfile)
        self._setup_model(ckptfile)

        if self.predict_cfg.output.exists():
            logger.warning(
                f"Output file {self.predict_cfg.output} already exists and will be overwritten"
            )

    def _setup_accelerator(self):
        if self.accelerator == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.accelerator == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device(self.accelerator.value)

        self.device = device

    def _setup_data(self, ckptfile: Path):
        self.datamodule = GenomeDataModule.from_pretrained(
            checkpoint_path=ckptfile,
            data_file=self.input_file,
            batch_size=self.batch_size,
            fragment_size=self.fragment_size,
            shuffle=False,
            lazy=self.lazy,
        )
        self.datamodule.setup("predict")
        if self.datamodule.dataset.any_multi_scaffold_genomes():
            self.graph_type = "scaffold"

        if (
            self.fragment_size is not None
            or self.datamodule.config.fragment_size != _SENTINEL_FRAGMENT_SIZE
        ):
            if self.graph_type == "scaffold":
                self.graph_type = "fragmented scaffold"
            else:
                self.graph_type = "fragment"

    def _setup_model(self, ckptfile: Path):
        self.model = self.model_type.from_pretrained(ckptfile).to(self.device).eval()
        self.is_genomic = isinstance(self.model, BaseProteinSetTransformer)

        if self.is_genomic:
            self.compute_scaffold_embeddings = (
                self._genomic_model_scaffold_embeddings
            )
        else:
            self.compute_scaffold_embeddings = (
                self._protein_model_scaffold_embeddings
            )

        expected_max_size = self.model.positional_embedding.max_size
        actual_max_size = self.datamodule.dataset.max_size
        if actual_max_size > expected_max_size:
            if not self.allow_genome_fragmenting:
                raise RuntimeError(
                    (
                        f"The maximum number of proteins in your dataset is {actual_max_size}"
                        f", but this model was trained with a max of {expected_max_size} "
                        "proteins. If you would like to proceed, set "
                        "`fragment_large_genomes=True` (or at the command line, set "
                        "--fragment_oversized_genomes true)"
                    )
                )

            if self.graph_type == "scaffold":
                # fragmenting scaffolds from multiscaffold genomes
                self.graph_type = "fragmented scaffold"
            else:
                # else all genomes are single scaffold, but we fragmented them
                self.graph_type = "fragment"
            self.datamodule.dataset.fragment(expected_max_size, inplace=True)

    def _create_earray(self, name: OutputType) -> tb.EArray:
        if name == "node":
            loc = "ctx_ptn"
            dim = self.model.encoder.out_dim
            expectedrows = self.datamodule.dataset.num_proteins
        elif name == "graph":
            loc = "data"
            dim = self.model.config.out_dim
            expectedrows = self.datamodule.dataset.num_scaffolds
        else:
            loc = "attn"
            dim = self.model.config.num_heads
            expectedrows = self.datamodule.dataset.num_proteins

        earray = self._file.create_earray(
            where=self._file.root,
            name=loc,
            atom=tb.Float32Atom(),
            shape=(0, dim),
            expectedrows=expectedrows,
            filters=self.filters,
        )
        return earray

    def _init_storage(self) -> dict["Predictor.OutputType", tb.EArray]:
        return {
            name: self._create_earray(name)
            for name in get_args(Predictor.OutputType)
        }

    def _init_in_memory_storage(
        self,
    ) -> dict["Predictor.OutputType", list[torch.Tensor]]:
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

        fragment_embeddings, scaffold_embeddings = (
            self._reduce_fragments_to_scaffolds(reduce=reduce)
        )

        # this is needed since the genome_label labels each individual FRAGMENT
        # so we need to convert it to labeling scaffolds
        # ie shape: [N fragments] -> [N scaffolds]
        scaffold_level_genome_label = convert_to_scaffold_level_genome_label(
            self.datamodule.dataset.scaffold_genome_label,
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
            scaffold_embeddings,
            self.datamodule.dataset.scaffold_genome_label,
            reduce=reduce,
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

        self._return_value["protein"] = node_embeddings

        # this attn is for pooling ptns within a scaffold
        # so if pst is a protein only model, it won't exist
        if self.model.is_genomic:
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
            self._file.rename_node(
                where=self._file.root, name="data", newname="genome"
            )
        else:
            self._file.remove_node(where=self._file.root, name="data")

    def _genomic_model_scaffold_embeddings(
        self, node_embeddings: torch.Tensor, batch: GenomeGraphBatch
    ) -> GraphAttnOutput:
        # if the model is a genomic level model, ie a subclass of BaseProteinSetTransformer,
        # then it has a decoder that returns the graph embeddings at the scaffold level

        return self.model.decoder(
            x=node_embeddings,
            ptr=batch.ptr,
            batch=batch.batch,
            return_attention_weights=True,
        )

    def _protein_model_scaffold_embeddings(
        self, node_embeddings: torch.Tensor, batch: GenomeGraphBatch
    ) -> GraphAttnOutput:
        # if the model is a protein level model, ie a subclass of BaseProteinSetTransformerEncoder,
        # then it does not have a decoder, so we just reduce the node embeddings to the scaffold level

        scaffold_embeddings = segment(node_embeddings, ptr=batch.ptr, reduce="mean")
        return GraphAttnOutput(out=scaffold_embeddings, attn=None)

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

            graph_output = self.compute_scaffold_embeddings(node_output.out, batch)

            if self.graph_embeddings:
                self.append(name="graph", data=graph_output.out)

                # don't include if not adding the protein embeddings
                if graph_output.attn is not None and self.node_embeddings:
                    self.append(name="attn", data=graph_output.attn)

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
            fragmented_embeddings, genome_embeddings = (
                self.reduce_fragmented_genomes()
            )
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
        self._cleanup_h5_file()

        self.close()

        if self.return_predictions:
            self._postprocess_in_memory_storage()
            self.in_memory_storage.clear()
            return self._return_value
        return None

    def close(self):
        self._file.close()
        self.datamodule.teardown("predict")

    def _cleanup_h5_file(self):
        for node in self._file.walk_nodes("/", classname="Array"):
            if node.shape[0] == 0:  # type: ignore
                self._file.remove_node(node)
