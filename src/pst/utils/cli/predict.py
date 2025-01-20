from pathlib import Path

from attrs import define, field

from pst.utils.attrs.validators import file_exists


@define
class PredictArgs:
    checkpoint: Path = field(validator=file_exists)
    """model checkpoint during inference"""

    outdir: Path = Path("output")
    """inference output directory"""

    fragment_oversized_genomes: bool = False
    """fragment oversized genomes that encode more proteins than the model was trained to 
    expect"""
