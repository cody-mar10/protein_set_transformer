import os
from pathlib import Path
from typing import Literal

import esm
import lightning as L
import torch
from attrs import define, field
from torch.utils.data import DataLoader

from pst.embed.data import SequenceDataset
from pst.embed.model import ESM2, ESM2Models
from pst.embed.writer import PredictionWriter
from pst.utils.attrs.dataclass_utils import AttrsDataclassUtilitiesMixin
from pst.utils.attrs.validators import positive_int


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


@define
class ModelArgs(AttrsDataclassUtilitiesMixin):
    esm: ESM2Models = ESM2Models.esm2_t6_8M
    """ESM-2 model key"""

    batch_size: int = field(default=1024, validator=positive_int)
    """batch size in number of tokens (amino acids)"""

    torch_hub: Path = _TORCH_HUB_PATH
    """path to the checkpoints/ directory with downloaded models"""


@define
class TrainerArgs(AttrsDataclassUtilitiesMixin):
    devices: int = field(default=1, validator=positive_int)
    """number of cpus/gpus to use"""

    accelerator: Literal["cpu", "gpu", "auto"] = "auto"
    """type of device to use"""

    precision: Literal[64, 32, 16, "bf16-mixed"] = 32
    """floating point precision"""


def embed(input: Path, outdir: Path, model_cfg: ModelArgs, trainer_cfg: TrainerArgs):
    """Embed protein sequences using ESM-2

    Args:
        input (Path): input protein fasta file with stop codons removed
        outdir (Path): output directory
        model_cfg (ModelArgs): MODEL
        trainer_cfg (TrainerArgs): TRAINER
    """
    outdir.mkdir(parents=True, exist_ok=True)

    torch.hub.set_dir(model_cfg.torch_hub)
    L.seed_everything(111)

    model = ESM2.from_model_name(model_cfg.esm)
    data = SequenceDataset(
        data=esm.FastaBatchedDataset.from_file(input),
        alphabet=model.alphabet,
        batch_size=model_cfg.batch_size,
    )
    writer = PredictionWriter(outdir=outdir, model=model, dataset=data)
    dataloader = DataLoader(
        dataset=data,
        # dataset is already pre-batched
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=SequenceDataset.collate_token_batches,
    )

    if trainer_cfg.accelerator == "auto":
        trainer_cfg.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if trainer_cfg.accelerator == "cpu":
        torch.set_num_threads(trainer_cfg.devices)
        trainer_cfg.precision = 32
        trainer_cfg.devices = 1

    trainer = L.Trainer(
        enable_checkpointing=False,
        callbacks=[writer],
        logger=False,
        **trainer_cfg.to_dict(),
    )

    trainer.predict(model=model, dataloaders=dataloader, return_predictions=False)
