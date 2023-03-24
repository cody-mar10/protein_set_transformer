import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger

import pst


def main():
    L.seed_everything(111)
    args = pst.utils.cli.parse_args()
    # TODO: should make subcommand mode ie fit/predict/test etc as arg

    logger = CSVLogger(**args.logger)
    datamodule = pst.data.GenomeSetDataModule(**args.data)
    data_dim = datamodule.feature_dimension
    model = pst.modules.GenomeTransformer(
        in_dim=data_dim, **args.model, **args.optimizer
    )
    if args.trainer["accelerator"] == "cpu":
        threads = args.trainer["devices"]
        torch.set_num_threads(threads)
        args.trainer["devices"] = 1

    args.trainer["precision"] = 16
    args.trainer["num_nodes"] = 1
    trainer = L.Trainer(logger=logger, **args.trainer)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
