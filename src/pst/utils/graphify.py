import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import tables as tb

from pst.data import dataset
from pst.data.utils import H5_FILE_COMPR_FILTERS
from pst.utils.cli.graphify import IOArgs, OptionalArgs

# silence the dataset logger which would report detection of multi-scaffold datasets
dataset.logger.setLevel("ERROR")
logger = logging.getLogger(__name__)


@dataclass
class FastaSummary:
    scaffold_sizes: defaultdict[str, int]
    strand: dict[str, int]
    genome_ids: dict[str, int]
    scaffold_to_genome: Optional[dict[str, str]]

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        scaffold_sizes = np.fromiter(self.scaffold_sizes.values(), dtype=int)
        strand = np.fromiter(self.strand.values(), dtype=int)

        if self.scaffold_to_genome is not None:
            genome_label = np.array(
                [
                    self.genome_ids[self.scaffold_to_genome[scaffold]]
                    for scaffold in self.scaffold_sizes.keys()
                ]
            )
        else:
            genome_label = None

        return scaffold_sizes, strand, genome_label


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


def summarize_fasta_file(
    fasta_file: Path,
    scaffold_map: dict[str, str],
    strand_map: dict[str, str],
) -> FastaSummary:
    strand_file_provided = bool(strand_map)
    _validate_fasta_file_headers(fasta_file, strand_file_provided)

    # this will be in the same order as the genomes appear in the FASTA file
    proteins_per_scaffold = defaultdict(int)
    strand_per_protein: dict[str, int] = dict()
    genome_ids: dict[str, int] = dict()
    detected_scaffold_to_genome: dict[str, str] = dict()

    # guaranteed to exist due to validation above
    strand_idx = 3
    curr_genome_id = 0
    for header in iter_proteins(fasta_file):
        fields = header.split(" # ")
        ptn = fields[0]

        ### STRAND
        strand = strand_map.get(ptn, None) or fields[strand_idx]
        strand = int(strand)
        strand_per_protein[ptn] = strand

        ### SCAFFOLD SIZES
        scaffold, ptn_id = ptn.rstrip().rsplit("_", 1)
        genome = scaffold_map.get(scaffold, scaffold)
        if genome not in genome_ids:
            genome_ids[genome] = curr_genome_id
            curr_genome_id += 1

        detected_scaffold_to_genome[scaffold] = genome
        proteins_per_scaffold[scaffold] += 1

    if all(
        scaffold == genome for scaffold, genome in detected_scaffold_to_genome.items()
    ):
        scaffold_to_genome = None
    else:
        scaffold_to_genome = detected_scaffold_to_genome

    return FastaSummary(
        scaffold_sizes=proteins_per_scaffold,
        strand=strand_per_protein,
        genome_ids=genome_ids,
        scaffold_to_genome=scaffold_to_genome,
    )


def validate(output: Path):
    try:
        # try to load a dataset
        dataset.GenomeDataset(output)
    except (RuntimeError, ValueError) as e:
        output.unlink()
        raise ValueError(
            f"There was a validation error when creating the graph-formatted dataset file {output}: {e}"
        ) from e


def single_file_to_graph_format(
    file: Path,
    fasta_file: Path,
    scaffold_map: dict[str, str],
    strand_map: dict[str, str],
    loc: str,
    output: Optional[Path] = None,
):
    logger.info(f"Processing {file} and {fasta_file}")
    fasta_summary = summarize_fasta_file(
        fasta_file=fasta_file,
        scaffold_map=scaffold_map,
        strand_map=strand_map,
    )

    scaffold_sizes, strand, genome_label = fasta_summary.to_numpy()

    ptr = np.concatenate(
        (
            [0],
            np.cumsum(scaffold_sizes),
        )
    )

    if output is None:
        datadir = file.parent
        output = datadir / f"{file.stem}.graphfmt.h5"

    with tb.open_file(file) as fsrc:
        ptn_embeddings = fsrc.root[loc][:]
        n_proteins = len(ptn_embeddings)

        if n_proteins != sum(scaffold_sizes):
            raise ValueError(
                "Number of proteins in the FASTA file does not match the number of protein embeddings"
            )

    with tb.open_file(output, "w") as fdst:
        locations = {
            "data": ptn_embeddings,
            "ptr": ptr,
            "strand": strand,
            "sizes": scaffold_sizes,
        }

        if genome_label is not None:
            locations["genome_label"] = genome_label

        for loc, data in locations.items():
            fdst.create_carray(
                where="/",
                name=loc,
                obj=data,
                filters=H5_FILE_COMPR_FILTERS,
            )

    validate(output)
    return output


def merge_graph_files(files: list[Path], output: Path):
    logger.info(f"Merging {len(files)} graph-formatted files into {output}")

    expected_fields = dataset.GenomeDataset.__minimum_h5_fields__ | {"genome_label"}
    all_data: dict[str, list[np.ndarray]] = {field: [] for field in expected_fields}
    for file in files:
        found_genome_label = False
        with tb.open_file(file) as fsrc:
            for field in expected_fields:
                try:
                    data = getattr(fsrc.root, field)
                except tb.NoSuchNodeError:
                    continue
                else:
                    if field == "genome_label":
                        found_genome_label = True
                    all_data[field].append(data[:])

        # this needs to happen after reading h5 file since we need to know the number of scaffolds
        if not found_genome_label:
            num_scaffolds = all_data["sizes"][-1].shape[0]
            genome_label = np.arange(num_scaffolds)
            all_data["genome_label"].append(genome_label)

    # now we need to merge all data
    final_fields: dict[str, np.ndarray] = {
        "data": np.vstack(all_data["data"]),
        "sizes": np.concatenate(all_data["sizes"]),
        "strand": np.concatenate(all_data["strand"]),
    }

    # ptr is the cumulative sum of scaffold sizes, so just recompute it
    total_scaffolds = final_fields["sizes"].shape[0]
    final_fields["ptr"] = np.zeros(total_scaffolds + 1, dtype=int)
    final_fields["ptr"][1:] = np.cumsum(final_fields["sizes"])

    # genome_label needs to be made relative to all data instead of each file
    final_genome_label = np.zeros(total_scaffolds, dtype=int)
    curr_max_genome_label = 0
    local_start = 0
    for genome_label in all_data["genome_label"]:
        num_scaffolds = genome_label.shape[0]
        local_end = local_start + num_scaffolds
        final_genome_label[local_start:local_end] = genome_label + curr_max_genome_label
        num_genomes = genome_label.max()
        curr_max_genome_label += num_genomes + 1

        local_start = local_end
    final_fields["genome_label"] = final_genome_label

    with tb.open_file(output, "w") as fdst:
        for loc, data in final_fields.items():
            fdst.create_carray(
                where="/",
                name=loc,
                obj=data,
                filters=H5_FILE_COMPR_FILTERS,
            )

    validate(output)

    for file in files:
        file.unlink()


def to_graph_format(io: IOArgs, optional: OptionalArgs):
    if optional.scaffold_map_file is not None:
        # scaffold -> genome mapping
        scaffold_map = tsv_to_dict(optional.scaffold_map_file)
    else:
        scaffold_map = dict()

    if optional.strand_file is not None:
        # protein -> strand mapping
        strand_map = tsv_to_dict(optional.strand_file)
    else:
        strand_map = dict()

    if io.multi_input_map is not None:
        outputs: list[Path] = []
        with io.multi_input_map.open() as fp:
            for line in fp:
                fasta_file, file = line.rstrip().split("\t")
                output = single_file_to_graph_format(
                    file=Path(file),
                    fasta_file=Path(fasta_file),
                    scaffold_map=scaffold_map,
                    strand_map=strand_map,
                    loc=io.loc,
                    output=None,
                )
                outputs.append(output)

        merge_graph_files(outputs, io.output or Path("combined_dataset.graphfmt.h5"))
    elif io.fasta_file is not None and io.file is not None:
        output = single_file_to_graph_format(
            file=io.file,
            fasta_file=io.fasta_file,
            scaffold_map=scaffold_map,
            strand_map=strand_map,
            loc=io.loc,
            output=io.output,
        )
        logging.info(f"Created graph-formatted dataset at {output}")
    else:
        raise ValueError(
            "Either --file AND --fasta-file must be provided OR --multi-input-map must be used"
        )
