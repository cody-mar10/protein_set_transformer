from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import esm
import lightning as L
import torch
from pydantic import BaseModel, Field, FilePath
from torch.utils.data import DataLoader

from pst.embed.data import SequenceDataset
from pst.embed.model import ESM2
from pst.embed.writer import PredictionWriter


def resolve_torch_hub_path() -> Path:
    """Resolve the path to the torch hub directory

    Returns:
        Path: path to torch hub directory
    """
    torch_hub = os.environ.get("TORCH_HOME", "")
    if torch_hub:
        return Path(torch_hub)

    xdg_dir = os.environ.get("XDG_CACHE_HOME", "")
    if xdg_dir:
        return Path(xdg_dir) / "torch/hub"

    return Path("~/.cache/torch/hub").expanduser()


_TORCH_HUB_PATH = resolve_torch_hub_path()


class ModelArgs(BaseModel):
    esm: ESM2.MODELVALUES = Field("esm2_t6_8M", description="ESM-2 model key")
    batch_size: int = Field(
        1024, description="batch size in number of tokens (amino acids)"
    )
    torch_hub: Path = Field(
        _TORCH_HUB_PATH,
        description="path to the checkpoints/ directory with downloaded models",
    )


class TrainerArgs(BaseModel):
    devices: int = Field(1, description="number of cpus/gpus to use")
    accelerator: Literal["cpu", "gpu", "auto"] = Field(
        "auto", description="type of device to use"
    )
    precision: Literal[64, 32, 16, "bf16"] = Field(
        default=32, description="floating point precision"
    )


class EmbedArgs(BaseModel):
    input: FilePath = Field(
        description="input protein fasta file, stop codons must be removed"
    )
    outdir: Path = Field(Path("output"), description="output directory")
    model: ModelArgs
    trainer: TrainerArgs


def embed(args: EmbedArgs):
    args.outdir.mkdir(parents=True, exist_ok=True)

    torch.hub.set_dir(args.model.torch_hub)
    L.seed_everything(111)

    model = ESM2.from_model_name(args.model.esm)
    data = SequenceDataset(
        data=esm.FastaBatchedDataset.from_file(args.input),
        alphabet=model.alphabet,
        batch_size=args.model.batch_size,
    )
    writer = PredictionWriter(outdir=args.outdir, model=model, dataset=data)
    dataloader = DataLoader(
        dataset=data,
        # dataset is already pre-batched
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=SequenceDataset.collate_token_batches,
    )

    if args.trainer.accelerator == "auto":
        args.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if args.trainer.accelerator == "cpu":
        torch.set_num_threads(args.trainer.devices)
        args.trainer.precision = 32
        args.trainer.devices = 1

    trainer = L.Trainer(
        enable_checkpointing=False,
        callbacks=[writer],
        logger=False,
        **args.trainer.model_dump(),
    )

    trainer.predict(model=model, dataloaders=dataloader, return_predictions=False)
