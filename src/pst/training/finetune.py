from __future__ import annotations

import logging

from pst.data.dataset import _SENTINEL_FRAGMENT_SIZE
from pst.data.modules import GenomeDataModule
from pst.training.utils.lightning import init_lightning_trainer
from pst.utils.auto import auto_resolve_model_type
from pst.utils.cli.modes import FinetuningMode

logger = logging.getLogger(__name__)


def finetune(args: FinetuningMode):
    logger.info("Finetuning a pretrained genomic PST with new data.")

    # if --fragment-size passed, dataset will be automatically fragmented
    datamodule = GenomeDataModule.from_pretrained(
        checkpoint_path=args.finetuning.checkpoint,
        data_file=args.data.file,
        command_line_config=args.data,  # allow updating batch size and fragment size from cli if set
        shuffle=True,
    )

    model_type = auto_resolve_model_type(args.experiment.pst_model_type)
    model = model_type.from_pretrained(args.finetuning.checkpoint)

    # need to check how many proteins are encoded in new data and fragment the
    # dataset accordingly
    expected_max_size = model.positional_embedding.max_size
    actual_max_size = datamodule.dataset.max_size

    if actual_max_size > expected_max_size:
        if not args.finetuning.fragment_oversized_genomes:
            raise RuntimeError(
                (
                    f"The maximum number of proteins in your dataset is {actual_max_size}"
                    f", but this model was trained with a max of {expected_max_size} "
                    "proteins. If you would like to proceed, pass --fragment-oversized-genomes "
                    "at the command line to automaticall fragment the dataset to the max size "
                    "that can be handled by the model. Or you can manually pass --fragment-size "
                    "to fragment the dataset into smaller chunks."
                )
            )

        if args.data.fragment_size != _SENTINEL_FRAGMENT_SIZE:
            # if allowed, fragment scaffolds into chunks that can be handled by the models
            # positional embeddding LUT
            logger.info(
                f"Fragmenting dataset into genomic fragments of {expected_max_size} proteins. "
                "If you want to use a different fragment size, pass --fragment-size at the command "
                "line."
            )
            datamodule.dataset.fragment(expected_max_size, inplace=True)
            # otherwise, if passing --fragment-size, the dataset will automatically handle its
            # own fragmentation prior to this

    trainer = init_lightning_trainer(args, checkpoint=True, early_stopping=True)
    trainer.fit(model=model, datamodule=datamodule)
