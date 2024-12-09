from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictArgs:
    checkpoint: Path
    """model checkpoint during inference"""

    outdir: Path = Path("output")
    """inference output directory"""

    fragment_oversized_genomes: bool = False
    """fragment oversized genomes that encode more proteins than the model was trained to 
    expect"""
