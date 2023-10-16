from __future__ import annotations

import sys
from typing import Any, Literal, Optional, cast, get_args

import tables as tb
import torch
from tqdm import tqdm

from pst.data.modules import DataConfig, GenomeDataModule
from pst.nn.modules import ModelConfig
from pst.nn.modules import ProteinSetTransformer as PST
from pst.typing import EdgeAttnOutput, GenomeGraphBatch, GraphAttnOutput
from pst.utils.cli.modes import InferenceMode


def model_inference(
    config: InferenceMode,
    node_embeddings: bool = True,
    graph_embeddings: bool = True,
    return_predictions: bool = False,
):
    predictor = Predictor(config, node_embeddings, graph_embeddings, return_predictions)

    return predictor.predict_loop()


class Predictor:
    OutputType = Literal["node", "graph", "attn"]

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

        self.setup()

        self.filters = tb.Filters(complevel=4, complib="blosc:lz4")

        self._file = tb.File(self.config.predict.outdir.joinpath("predictions.h5"), "w")

        self.storage, self.in_memory_storage = self._init_storages()

    def setup(self):
        self._setup_accelerator()
        ckpt = torch.load(self.config.predict.checkpoint, map_location=self.device)

        self._setup_data(ckpt)
        self._setup_model(ckpt)

        self.config.predict.outdir.mkdir(parents=True, exist_ok=True)

    def _setup_accelerator(self):
        if self.config.trainer.accelerator == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.config.trainer.accelerator == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device(self.config.trainer.accelerator)

        self.device = device

    def _setup_data(self, ckpt: dict[str, Any]):
        data_config = DataConfig.model_construct(
            file=self.config.data.file,
            **ckpt["datamodule_hyper_parameters"],
        )

        self.datamodule = GenomeDataModule(data_config, shuffle=False)
        self.datamodule.setup("predict")

    def _setup_model(self, ckpt: dict[str, Any]):
        model_config = ModelConfig.model_validate(ckpt["hyper_parameters"])

        self.model = PST(model_config)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)

        expected_max_size = self.model.positional_embedding.max_size
        actual_max_size = self.datamodule.dataset.max_size
        if actual_max_size > expected_max_size:
            # TODO: the default behavior for other transformers is to just work with the
            # first n words that work with the model's max size... that is sort of
            # complicated here since that would require adjusting the genome's graph
            # representation, ie the edge_index

            raise RuntimeError(
                (
                    f"The maximum number of proteins in your dataset is {actual_max_size}"
                    f", but this model was trained with a max of {expected_max_size} "
                    "proteins."
                )
            )

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

    @torch.no_grad()
    def predict_loop(self) -> Optional[dict[Predictor.OutputType, list[torch.Tensor]]]:
        dataloader = self.datamodule.predict_dataloader()

        batch: GenomeGraphBatch
        for batch in tqdm(dataloader, file=sys.stdout):
            batch = batch.to(self.device)  # type: ignore

            pos_emb = self.model.positional_embedding(batch.pos.squeeze())
            strand_emb = self.model.strand_embedding(batch.strand)
            x_cat = self.model._concatenate_embeddings(
                batch.x, positional_embed=pos_emb, strand_embed=strand_emb
            )

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

        if self.return_predictions:
            return self.in_memory_storage
        return None
