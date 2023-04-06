from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import pst


def _train_main(args: pst.utils.cli.Args):
    checkpointing = ModelCheckpoint(monitor="val_loss", save_top_k=3, every_n_epochs=1)
    datamodule = pst.data.GenomeSetDataModule(**args.data, args=args)
    data_dim = datamodule.feature_dimension
    model = pst.modules.GenomeTransformer(
        in_dim=data_dim, **args.model, **args.optimizer
    )
    trainer = L.Trainer(
        callbacks=[checkpointing],
        logger=True,
        log_every_n_steps=5,
        **args.trainer,
    )
    trainer.fit(model=model, datamodule=datamodule)


def _test_main(args: pst.utils.cli.Args):
    pass


def _predict_main(args: pst.utils.cli.Args):
    model = pst.modules.GenomeTransformer.load_from_checkpoint(
        args.predict["checkpoint"]
    )

    datamodule = pst.data.GenomeSetDataModule(**args.data, args=args, stage="predict")
    writer = pst.utils.PredictionWriter(
        outdir=args.predict["outdir"],
        dataset=datamodule,
    )
    trainer = L.Trainer(
        callbacks=[writer],
        **args.trainer,
    )
    trainer.predict(model=model, datamodule=datamodule)


def _simple_data(
    args: pst.utils.cli.Args,
) -> tuple[pst.data.SimpleGenomeDataset, DataLoader]:
    dataset = pst.data.SimpleGenomeDataset(
        data_file=args.data["data_file"], genome_metadata=args.data["metadata_file"]
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.data["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_batch,
    )
    return dataset, dataloader


def _debug_main(args: pst.utils.cli.Args):
    dataset, dataloader = _simple_data(args)

    data_dim = dataset._data.shape[-1]
    model = pst.modules.GenomeTransformer(
        in_dim=data_dim, use_scheduler=False, **args.model, **args.optimizer
    )
    args.trainer["max_epochs"] = 10
    trainer = L.Trainer(
        logger=True,
        detect_anomaly=True,
        overfit_batches=10,
        **args.trainer,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)


def _precompute_main(args: pst.utils.cli.Args):
    dataset, dataloader = _simple_data(args)
    precompute_sampler = pst.sampling.PrecomputeSampler(
        data_file=args.data["data_file"],
        batch_size=args.data["batch_size"],
        dataloader=dataloader,
        sample_rate=args.model["sample_rate"],
        scale=args.model["sample_scale"],
        device=args.trainer["accelerator"],
    )
    precompute_sampler.save()


def main():
    L.seed_everything(111)
    print("Lightning version: ", L.__version__)
    args = pst.utils.cli.parse_args()

    if args.trainer["accelerator"] == "cpu":
        threads = args.trainer["devices"]
        torch.set_num_threads(threads)
        args.trainer["devices"] = 1
    elif args.trainer["accelerator"] == "gpu":
        args.trainer["precision"] = "16-mixed"
        args.trainer["num_nodes"] = 1

    if args.mode == "train":
        _train_main(args)
    elif args.mode == "predict":
        _predict_main(args)
    elif args.mode == "debug":
        _debug_main(args)
    elif args.mode == "precompute":
        _precompute_main(args)
    else:
        _test_main(args)


if __name__ == "__main__":
    main()
