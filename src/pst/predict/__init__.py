from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional, cast

import lightning as L
import tables as tb
import torch
from tqdm import tqdm

from pst.data.modules import DataConfig, GenomeDataModule
from pst.nn.modules import ModelConfig
from pst.nn.modules import ProteinSetTransformer as PST
from pst.predict.writer import PredictionWriter
from pst.typing import GenomeGraphBatch, PairTensor
from pst.utils.cli.modes import InferenceMode
from pst.utils.cli.trainer import TrainerArgs

GraphOutputType = list[PairTensor]
OptionalGraphOutput = Optional[GraphOutputType]

NodeOutputType = list[torch.Tensor]
OptionalNodeOutput = Optional[NodeOutputType]


def model_inference(
    config: InferenceMode,
    node_embeddings: bool = True,
    graph_embeddings: bool = True,
    return_predictions: bool = False,
):
    if not node_embeddings and not graph_embeddings:
        raise ValueError(
            "At least one of node_embeddings and graph_embeddings must be True"
        )

    if config.trainer.accelerator == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.trainer.accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device(config.trainer.accelerator)

    ckpt: dict[str, Any] = torch.load(
        config.predict.checkpoint,
        map_location=device,
    )

    model_config = ModelConfig.model_validate(ckpt["hyper_parameters"])

    model = PST(model_config)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    data_config = DataConfig.model_construct(
        file=config.data.file,
        **ckpt["datamodule_hyper_parameters"],
    )

    datamodule = GenomeDataModule(data_config)
    datamodule.setup("predict")

    expected_max_size = model.positional_embedding.max_size
    actual_max_size = datamodule.dataset.max_size
    if actual_max_size > expected_max_size:
        # TODO: the default behavior for other transformers is to just work with the first
        # n words that work with the model's max size... that is sort of complicated here
        # since that would require adjusting the genome's graph representation,
        # ie the edge_index

        raise RuntimeError(
            (
                f"The maximum number of proteins in your dataset is {actual_max_size}, "
                f"but this model was trained with a max of {expected_max_size} proteins."
            )
        )

    config.predict.outdir.mkdir(parents=True, exist_ok=True)

    if graph_embeddings and not node_embeddings:
        return _model_inference_graph_embeddings(
            model,
            datamodule,
            config.predict.outdir,
            config.trainer,
            return_predictions,
        )
    elif node_embeddings and not graph_embeddings:
        return _model_inference_node_embeddings(
            model,
            datamodule,
            config.predict.outdir,
            return_predictions,
        )
    elif node_embeddings and graph_embeddings:
        return _fused_inference(
            model,
            datamodule,
            config.predict.outdir,
        )


def _model_inference_graph_embeddings(
    model: PST,
    datamodule: GenomeDataModule,
    outdir: Path,
    trainer_config: TrainerArgs,
    return_predictions: bool = False,
) -> OptionalGraphOutput:
    writer = PredictionWriter(
        outdir=outdir,
        model=model,
        datamodule=datamodule,
    )

    trainer = L.Trainer(
        callbacks=[writer],
        max_time=trainer_config.max_time.value,
        **trainer_config.model_dump(exclude={"max_time"}),
    )

    output = trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=return_predictions,
    )
    output = cast(OptionalGraphOutput, output)
    if return_predictions:
        return output
    return None


# TODO: extract same details as fused step into fns
def _model_inference_node_embeddings(
    model: PST,
    datamodule: GenomeDataModule,
    outdir: Path,
    return_predictions: bool = False,
) -> OptionalNodeOutput:
    # have to do everything manually here it seems

    model.eval()
    dataloader = datamodule.predict_dataloader()
    filters = tb.Filters(complevel=4, complib="blosc:lz4")

    predictions: list[torch.Tensor] = list()
    with ExitStack() as ctx:
        _no_grad = ctx.enter_context(torch.no_grad())
        fp: tb.File = ctx.enter_context(tb.File(outdir.joinpath("predictions.h5"), "w"))

        n = datamodule.dataset.data.shape[0]
        array = fp.create_earray(
            fp.root,
            "ctx_ptn",
            atom=tb.Float32Atom(),
            shape=(0, model.encoder.out_dim),
            expectedrows=n,
            filters=filters,
        )

        batch: GenomeGraphBatch
        for batch in tqdm(dataloader, file=sys.stdout):
            batch = batch.to(model.device)  # type: ignore
            node_embeddings: torch.Tensor = model.encoder(
                batch.x, batch.edge_index, batch.batch
            )

            _append_to_earray(array, node_embeddings)

            if return_predictions:
                predictions.append(node_embeddings.detach().cpu())

    if return_predictions:
        return predictions
    return None


def _fused_inference(
    model: PST,
    datamodule: GenomeDataModule,
    outdir: Path,
):
    # have to do everything manually here it seems
    model = model.to()
    model.eval()
    dataloader = datamodule.predict_dataloader()
    filters = tb.Filters(complevel=4, complib="blosc:lz4")

    # TODO: return predictions
    with ExitStack() as ctx:
        _no_grad = ctx.enter_context(torch.no_grad())
        fp: tb.File = ctx.enter_context(tb.File(outdir.joinpath("predictions.h5"), "w"))

        n = datamodule.dataset.data.shape[0]
        node_array = fp.create_earray(
            fp.root,
            "ctx_ptn",
            atom=tb.Float32Atom(),
            shape=(0, model.encoder.out_dim),
            expectedrows=n,
            filters=filters,
        )

        graph_data_array = fp.create_earray(
            fp.root,
            name="data",
            atom=tb.Float32Atom(),
            shape=(0, model.config.out_dim),
            expectedrows=datamodule.dataset.sizes.numel(),
            filters=filters,
        )

        graph_attn_array = fp.create_earray(
            fp.root,
            name="attn",
            atom=tb.Float32Atom(),
            shape=(0, model.config.num_heads),
            expectedrows=n,
            filters=filters,
        )

        batch: GenomeGraphBatch
        for batch in tqdm(dataloader, file=sys.stdout):
            batch = batch.to(model.device)  # type: ignore

            # this is technically after the final norm too
            node_embeddings: torch.Tensor = model.encoder(
                batch.x, batch.edge_index, batch.batch
            )
            _append_to_earray(node_array, node_embeddings)

            graph_embeddings: torch.Tensor
            attn: torch.Tensor
            graph_embeddings, attn = model.decoder(
                x=node_embeddings,
                ptr=batch.ptr,
                batch=batch.batch,
                return_attention_weights=True,
            )

            _append_to_earray(graph_data_array, graph_embeddings)
            _append_to_earray(graph_attn_array, attn)


def _append_to_earray(earray: tb.EArray, data: torch.Tensor):
    earray.append(data.detach().cpu().numpy())
