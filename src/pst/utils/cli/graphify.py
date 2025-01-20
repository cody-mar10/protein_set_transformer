from pathlib import Path
from typing import Optional

from attrs import define, field

from pst.utils.attrs.validators import optionally_existing_file


@define
class IOArgs:
    """IO"""

    file: Optional[Path] = field(default=None, validator=optionally_existing_file)
    """ESM2 protein embeddings in .h5 format. Either this AND --io.fasta_file must be provided OR 
    --io.multi_input_map must be used."""

    loc: str = "data"
    """location of protein embeddings in .h5 file relative to the root node"""

    fasta_file: Optional[Path] = field(default=None, validator=optionally_existing_file)
    """The protein FASTA file used to generate the protein embeddings. The embeddings MUST be in
    the same order as this file. The protein headers MUST be prodigal format: '>scaffold_ptn#'.
    Either this AND --io.file must be provided OR --io.multi_input_map must be used."""

    multi_input_map: Optional[Path] = field(
        default=None, validator=optionally_existing_file
    )
    """tab-delimited mapping file for inputting multiple embedding and fasta files. Order of
    columns (no header): (FASTA file, ESM embedding file). Either this or --io.file AND --io.fasta_file
    must be provided."""

    output: Optional[Path] = None
    """output file name. Defaults to the input file name with '.graphfmt' added before the .h5
    extension"""

    def __attrs_post_init__(self):
        if (
            self.file is None
            and self.fasta_file is None
            and self.multi_input_map is None
        ):
            raise ValueError(
                "Either --io.file AND --io.fasta_file must be provided OR --io.multi_input_map must be used"
            )

        elif self.file is not None and self.fasta_file is None:
            raise ValueError(
                "If --io.file is provided, --io.fasta_file must also be provided"
            )
        elif self.file is None and self.fasta_file is not None:
            raise ValueError(
                "If --io.fasta_file is provided, --io.file must also be provided"
            )
        elif self.multi_input_map is not None and (
            self.file is not None or self.fasta_file is not None
        ):
            raise ValueError(
                "If --io.multi_input_map is provided, NEITHER --io.file NOR --io.fasta_file should be "
                "provided"
            )


@define
class OptionalArgs:
    """OPTIONAL"""

    strand_file: Optional[Path] = field(
        default=None, validator=optionally_existing_file
    )
    """tab-delimited strand file for each embedded protein. Must be in the same order as the 
    FASTA file. Order of columns (no header): (protein name, strand [-1, 1])."""

    scaffold_map_file: Optional[Path] = field(
        default=None, validator=optionally_existing_file
    )
    """tab-delimited mapping file for multi-scaffold viruses Order of columns (no header): 
    (scaffold name, genome name). See FASTA file description to understand how we define the 
    scaffold name."""
