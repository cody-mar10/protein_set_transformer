from pathlib import Path

from attrs import define, field

from pst.utils.attrs.validators import file_exists


@define
class FinetuningArgs:
    checkpoint: Path = field(validator=file_exists)
    """pre-trained model checkpoint"""

    outdir: Path = Path("output")
    """finetuning output directory"""

    fragment_oversized_genomes: bool = False
    """fragment oversized genomes up to the max size of the model's positional embedding LUT"""
