from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinetuningArgs:
    checkpoint: Path
    """pre-trained model checkpoint"""

    outdir: Path = Path("output")
    """finetuning output directory"""

    fragment_oversized_genomes: bool = False
    """fragment oversized genomes up to the max size of the model's positional embedding LUT"""
