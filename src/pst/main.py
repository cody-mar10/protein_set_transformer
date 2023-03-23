import lightning as L
import torch

import pst


def main():
    L.seed_everything(111)
    args = pst.utils.cli.parse_args()
    # TODO: should make subcommand mode ie fit/predict/test etc as arg
    model = pst.modules.GenomeTransformer(**args.model, **args.optimizer)
    datamodule = pst.data.GenomeSetDataModule(**args.data)
    if args.trainer["accelerator"] == "cpu":
        threads = args.trainer["devices"]
        torch.set_num_threads(threads)
        args.trainer["devices"] = 1
    trainer = L.Trainer(**args.trainer)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
