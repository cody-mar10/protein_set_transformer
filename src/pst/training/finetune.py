import logging
from pathlib import Path
from typing import Optional

from pst.data.modules import GenomeDataModule
from pst.nn.base import BaseModelTypes
from pst.nn.modules import ProteinSetTransformer
from pst.training.utils.lightning import init_lightning_trainer
from pst.training.utils.pst import _add_group_weights
from pst.utils.cli.finetune import FinetuningArgs
from pst.utils.cli.trainer import TrainerArgs

logger = logging.getLogger(__name__)


class FinetuneMode:
    def finetune(
        self,
        file: Path,
        finetuning: FinetuningArgs,
        trainer: TrainerArgs,
        model_type: BaseModelTypes = ProteinSetTransformer,
        batch_size: Optional[int] = None,
        fragment_size: Optional[int] = None,
        lazy: bool = False,
    ):
        """Finetune a pretrained PST with new data.

        Args:
            file (FilePath): The path to the graph-formatted .h5 data file.
            finetuning (FinetuningArgs): FINETUNING
            trainer (TrainerArgs): TRAINER
            model_type (BaseModelTypes, optional): PST model type to use for prediction.
            batch_size (Optional[int]): The batch size to use for finetuning if different from
                the checkpoint.
            fragment_size (Optional[int]): The genome fragmentation size if different from the
                checkpoint.
            lazy (bool): Whether to use lazy loading for the dataset. If True, will load data lazily
                from the file. If False, will fully load data into memory. This is useful for large
                datasets that may not fit into memory. Defaults to False.
        """
        logger.info(f"Finetuning a pretrained PST of type {model_type.__name__} with new data.")

        # if --fragment-size passed, dataset will be automatically fragmented
        # NOTE: don't need to worry about previous run having validation
        # bc validation is not saved as an hparam checkpoint
        datamodule = GenomeDataModule.from_pretrained(
            checkpoint_path=finetuning.checkpoint,
            data_file=file,
            batch_size=batch_size,
            fragment_size=fragment_size,
            shuffle=True,
            lazy=lazy,
        )

        if model_type is ProteinSetTransformer:
            # add group weights to the datamodule
            _add_group_weights(datamodule)

        model = model_type.from_pretrained(finetuning.checkpoint)

        # need to check how many proteins are encoded in new data and fragment the dataset accordingly
        expected_max_size = model.positional_embedding.max_size
        actual_max_size = datamodule.dataset.max_size

        if actual_max_size > expected_max_size:
            if not finetuning.fragment_oversized_genomes:
                msg = (
                    f"The maximum number of proteins in your dataset is {actual_max_size} but "
                    f"this model was trained with a max of {expected_max_size} proteins. If "
                    "you would like to proceed, pass --fragment_oversized_genomes true at the "
                    "command line to automatically fragment the dataset to the max size that "
                    "can be handled by the model. Or you can manually pass --fragment_size to "
                    "fragment the dataset into smaller chunks."
                )
                raise RuntimeError(msg)

            if fragment_size is None:
                # if allowed, fragment scaffolds into chunks that can be handled by the models
                # positional embeddding LUT
                logger.info(
                    f"Fragmenting dataset into genomic fragments of {expected_max_size} "
                    "proteins. If you want to use a different fragment size, pass "
                    "--fragment_size at the command line."
                )
                datamodule.dataset.fragment(expected_max_size, inplace=True)
                # otherwise, if passing --fragment-size, the dataset will automatically handle its
                # own fragmentation prior to this

        trainer.default_root_dir = finetuning.outdir
        trainer_obj = init_lightning_trainer(
            model_cfg=model.config,
            trainer_cfg=trainer,
            checkpoint=True,
            early_stopping=True,
        )
        trainer_obj.fit(model=model, datamodule=datamodule)
