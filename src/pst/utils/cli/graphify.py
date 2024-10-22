from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, FilePath


class IOArgs(BaseModel):
    file: Optional[FilePath] = Field(
        None,
        description=(
            "ESM2 protein embeddings in .h5 format. Either this AND --fasta-file must be "
            "provided OR --multi-input-map must be used."
        ),
    )
    loc: str = Field(
        "data",
        description="location of protein embeddings in .h5 file relative to the root node",
    )
    fasta_file: Optional[FilePath] = Field(
        None,
        description=(
            "The protein FASTA file used to generate the protein embeddings. The embeddings MUST "
            "be in the same order as this file. The protein headers MUST be prodigal format: "
            "'>scaffold_ptn#'. Either this AND --file must be provided OR --multi-input-map must "
            "be used."
        ),
    )
    multi_input_map: Optional[FilePath] = Field(
        None,
        description=(
            "tab-delimited mapping file for inputting multiple embedding and fasta files. "
            "Order of columns (no header): (FASTA file, ESM embedding file). Either this or "
            "--file AND --fasta-file must be provided."
        ),
    )
    output: Optional[Path] = Field(
        None,
        description=(
            "output file name. Defaults to the input file name with '.graphfmt' added before "
            "the .h5 extension"
        ),
    )


class OptionalArgs(BaseModel):
    strand_file: Optional[Path] = Field(
        None,
        description=(
            "tab-delimited strand file for each embedded protein. Must be in the same order as "
            "the FASTA file. Order of columns (no header): (protein name, strand [-1, 1])."
        ),
    )
    scaffold_map_file: Optional[Path] = Field(
        None,
        description=(
            "tab-delimited mapping file for multi-scaffold viruses Order of columns (no header): "
            "(scaffold name, genome name). See FASTA file description to understand how we "
            "define the scaffold name."
        ),
    )


class GraphifyArgs(BaseModel):
    inputs: IOArgs
    optional: OptionalArgs
