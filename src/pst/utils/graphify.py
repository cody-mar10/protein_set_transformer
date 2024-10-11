from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import tables as tb

from pst.data.utils import H5_FILE_COMPR_FILTERS
from pst.utils.cli.graphify import GraphifyArgs


@dataclass
class FastaSummary:
    genome_sizes: defaultdict[str, int]
    strand: dict[str, int]

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        genome_sizes = np.fromiter(self.genome_sizes.values(), dtype=int)
        strand = np.fromiter(self.strand.values(), dtype=int)

        return genome_sizes, strand


def tsv_to_dict(tsv_file: Path) -> dict[str, str]:
    with tsv_file.open("rt") as fp:
        return dict(tuple(line.rstrip().split("\t")) for line in fp)  # type: ignore


def iter_proteins(fasta_file: Path) -> Iterator[str]:
    with fasta_file.open() as fp:
        for line in fp:
            if line.startswith(">"):
                ptn = line[1:].rstrip()
                yield ptn


def _validate_fasta_file_headers(fasta_file: Path, strand_file_provided: bool):
    header = next(iter_proteins(fasta_file))

    if not strand_file_provided and " # " not in header:
        raise ValueError(
            "FASTA file headers must be in prodigal format: '>scaffold_ptn#' with the additional metadata separated by ' # '"
        )

    ptn = header.split(" # ")[0]
    if "_" not in ptn:
        raise ValueError(
            "FASTA file headers must be in prodigal format: '>scaffold_ptn#'"
        )


def _head(d: dict) -> dict:
    from itertools import islice

    return dict(islice(d.items(), 5))


def summarize_fasta_file(
    fasta_file: Path,
    scaffold_map_file: Optional[Path] = None,
    strand_file: Optional[Path] = None,
) -> FastaSummary:
    strand_file_provided = strand_file is not None
    _validate_fasta_file_headers(fasta_file, strand_file_provided)

    if scaffold_map_file is not None:
        # scaffold -> genome mapping
        scaffold_map = tsv_to_dict(scaffold_map_file)
    else:
        scaffold_map = dict()

    if strand_file_provided:
        # protein -> strand mapping
        strand_map = tsv_to_dict(strand_file)
    else:
        strand_map = dict()

    # this will be in the same order as the genomes appear in the FASTA file
    proteins_per_genome = defaultdict(int)
    strand_per_protein: dict[str, int] = dict()

    # guaranteed to exist due to validation above
    strand_idx = 3
    for header in iter_proteins(fasta_file):
        fields = header.split(" # ")
        ptn = fields[0]

        ### STRAND
        strand = strand_map.get(ptn, None) or fields[strand_idx]
        strand = int(strand)
        strand_per_protein[ptn] = strand

        ### GENOME SIZES
        scaffold, ptn_id = ptn.rstrip().rsplit("_", 1)
        genome = scaffold_map.get(scaffold, scaffold)
        proteins_per_genome[genome] += 1

    return FastaSummary(genome_sizes=proteins_per_genome, strand=strand_per_protein)


def to_graph_format(args: GraphifyArgs):
    fasta_summary = summarize_fasta_file(
        args.inputs.fasta_file,
        args.optional.scaffold_map_file,
        args.optional.strand_file,
    )

    genome_sizes, strand = fasta_summary.to_numpy()

    ptr = np.concatenate(([0], np.cumsum(genome_sizes)))

    if args.inputs.output is None:
        datadir = args.inputs.file.parent
        output = datadir / f"{args.inputs.file.stem}.graphfmt.h5"
    else:
        output = args.inputs.output

    with tb.open_file(args.inputs.file) as fsrc:
        ptn_embeddings = fsrc.root[args.inputs.loc][:]
        n_proteins = len(ptn_embeddings)

        if n_proteins != sum(genome_sizes):
            raise ValueError(
                "Number of proteins in the FASTA file does not match the number of protein embeddings"
            )

    with tb.open_file(output, "w") as fdst:
        locations = {
            "data": ptn_embeddings,
            "ptr": ptr,
            "strand": strand,
            "sizes": genome_sizes,
        }

        for loc, data in locations.items():
            fdst.create_carray(
                where="/",
                name=loc,
                obj=data,
                filters=H5_FILE_COMPR_FILTERS,
            )
