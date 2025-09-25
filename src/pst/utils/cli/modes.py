from pathlib import Path

from pst.embed.main import ModelArgs as EmbedModelArgs
from pst.embed.main import TrainerArgs as EmbedTrainerArgs
from pst.embed.main import embed
from pst.utils.cli import download, graphify
from pst.utils.download import DryadDownloader
from pst.utils.graphify import to_graph_format


class EmbedMode:
    def embed(
        self,
        input: Path,
        outdir: Path,
        model: EmbedModelArgs,
        trainer: EmbedTrainerArgs,
    ):
        """Embed protein sequences using ESM-2

        Args:
            input (Path): input protein fasta file with stop codons removed
            outdir (Path): output directory
            model (ModelArgs): MODEL
            trainer (TrainerArgs): TRAINER
        """
        embed(input, outdir, model, trainer)


class GraphifyMode:
    def graphify(self, io: graphify.IOArgs, optional: graphify.OptionalArgs):
        """Pre-processing mode to convert raw ESM2 protein embeddings into a graph-formatted
        dataset for use with other PST modes."""
        to_graph_format(io, optional)


class DownloadMode:
    def download(
        self,
        manuscript: download.ManuscriptDataArgs,
        cluster: download.ClusterArgs,
        model: download.ModelArgs,
        embeddings: download.EmbeddingsArgs,
        all: bool = False,
        outdir: Path = Path("pstdata"),
    ):
        """Download mode to download data and trained models from DRYAD. Example usage: pst download --manuscript.choices="[source_data, supplementary_data]"

        Args:
            manuscript (ManuscriptDataArgs): MANUSCRIPT DATA
            cluster (ClusterArgs): CLUSTER DATA
            model (ModelArgs): TRAINED MODELS
            embeddings (EmbeddingsArgs): EMBEDDINGS
            all (bool, optional): Download all data from the DRYAD repository.
            outdir (Path, optional): Output directory to save files.
        """

        downloader = DryadDownloader(
            manuscript=manuscript,
            cluster=cluster,
            model=model,
            embeddings=embeddings,
            all=all,
            outdir=outdir,
        )
        downloader.download()
