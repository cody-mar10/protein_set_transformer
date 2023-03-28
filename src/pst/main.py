import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

import pst


def main():
    L.seed_everything(111)
    print("Lightning version: ", L.__version__)
    args = pst.utils.cli.parse_args()
    # TODO: should make subcommand mode ie fit/predict/test etc as arg
    checkpointing = ModelCheckpoint(monitor="val_loss", save_top_k=3, every_n_epochs=1)
    datamodule = pst.data.GenomeSetDataModule(**args.data)
    data_dim = datamodule.feature_dimension
    model = pst.modules.GenomeTransformer(
        in_dim=data_dim, **args.model, **args.optimizer
    )
    if args.trainer["accelerator"] == "cpu":
        threads = args.trainer["devices"]
        torch.set_num_threads(threads)
        args.trainer["devices"] = 1
    elif args.trainer["accelerator"] == "gpu":
        args.trainer["precision"] = "16-mixed"
        args.trainer["num_nodes"] = 1

    trainer = L.Trainer(
        callbacks=[checkpointing],
        logger=True,
        **args.trainer,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
