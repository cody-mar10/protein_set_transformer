import gzip
import shutil
import tarfile
import zipfile
from functools import partial
from pathlib import Path

import requests
from tqdm import tqdm

from pst.utils.cli.download import (
    ClusterArgs,
    EmbeddingsArgs,
    ManuscriptDataArgs,
    ModelArgs,
)

_DataclassT = ClusterArgs | EmbeddingsArgs | ManuscriptDataArgs | ModelArgs

choice_mapper = {
    "manuscript": {
        "source_data": "source_data.tar.gz",
        "supplementary_data": "supplementary_data.tar.gz",
        "supplementary_tables": "supplementary_tables.tar.gz",
        "host_prediction": "host_prediction.tar.gz",
        "fasta": "fasta.tar.gz",
        "foldseek_databases": "foldseek_databases.tar.gz",
        "README": "README.md",
    },
    "cluster": {
        "genome": "genome_clusters.tar.gz",
        "protein": "protein_clusters.tar.gz",
    },
    "model": {
        "PST-TL-P__small": "PST-TL-P__small.ckpt.gz",
        "PST-TL-P__large": "PST-TL-P__large.ckpt.gz",
        "PST-TL-T__small": "PST-TL-T__small.ckpt.gz",
        "PST-TL-T__large": "PST-TL-T__large.ckpt.gz",
        "PST-MLM": "PST-MLM.tar.gz",
    },
    "embeddings": {
        ### protein:
        "esm2": "esm_embeddings.tar.gz",
        "IMGVR_PST-TL-P__large": "IMGVRv4_test_set_PST-TL-P__large_protein_embeddings.h5",
        "IMGVR_PST-TL-P__small": "IMGVRv4_test_set_PST-TL-P__small_protein_embeddings.h5",
        "IMGVR_PST-TL-T__large": "IMGVRv4_test_set_PST-TL-T__large_protein_embeddings.h5",
        "IMGVR_PST-TL-T__small": "IMGVRv4_test_set_PST-TL-T__small_protein_embeddings.h5",
        "MGnify_PST-TL-P__large": "MGnify_test_set_PST-TL-P__large_protein_embeddings.h5",
        "MGnify_PST-TL-P__small": "MGnify_test_set_PST-TL-P__small_protein_embeddings.h5",
        "MGnify_PST-TL-T__large": "MGnify_test_set_PST-TL-T__large_protein_embeddings.h5",
        "MGnify_PST-TL-T__small": "MGnify_test_set_PST-TL-T__small_protein_embeddings.h5",
        "genslm_ORF": "genslm_ORF_embeddings.h5",
        "train_PST-TL-P__large": "PST_training_set_PST-TL-P__large_protein_embeddings.h5",
        "train_PST-TL-P__small": "PST_training_set_PST-TL-P__small_protein_embeddings.h5",
        "train_PST-TL-T__large": "PST_training_set_PST-TL-T__large_protein_embeddings.h5",
        "train_PST-TL-T__small": "PST_training_set_PST-TL-T__small_protein_embeddings.h5",
        ### genome:
        "PST-TL_genome": "PST-TL_genome_embeddings.tar.gz",
        "other_genome": "other_genome_embeddings.tar.gz",
    },
}


class DryadDownloader:
    # dataset doi
    DRYAD_DOI_REST_API = r"doi%3A10.5061%2Fdryad.d7wm37q8w"
    DRYAD_REST_API = "https://datadryad.org/api/v2"

    def __init__(
        self,
        manuscript: ManuscriptDataArgs,
        cluster: ClusterArgs,
        model: ModelArgs,
        embeddings: EmbeddingsArgs,
        all: bool,
        outdir: Path,
    ):
        self.args: dict[str, _DataclassT] = {
            "manuscript": manuscript,
            "cluster": cluster,
            "model": model,
            "embeddings": embeddings,
        }

        self.download_all = all

        self._validate_downloads()

        self.outdir = outdir

        self._session: requests.Session | None = None

    def _validate_downloads(self):
        if self.download_all:
            return

        for dc in self.args.values():
            if dc.choices is not None:
                return

        raise ValueError(
            "No data to download. Please specify at least one download."
        )

    def get_files_to_download(self) -> list[str]:
        files: list[str] = list()

        if self.download_all:
            for filemap in choice_mapper.values():
                files.extend(filemap.values())

        else:
            for name, dc in self.args.items():
                mapper = choice_mapper[name]

                choices = dc.choices
                if choices is not None:
                    for choice in choices:
                        file = mapper[choice]
                        files.append(file)

            # unique files only
            files = list(set(files))

        return files

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()

        return self._session

    def get_dryad_dataset_id(self) -> int:
        # this will always point to the latest version
        url = f"{self.DRYAD_REST_API}/datasets/{self.DRYAD_DOI_REST_API}"

        response = self.session.get(url)
        dryad_dataset_id = int(
            response.json()["_links"]["stash:version"]["href"].split("/")[-1]
        )

        return dryad_dataset_id

    def get_dryad_file_ids(self, dataset_id: int) -> dict[str, int]:
        baseurl = f"{self.DRYAD_REST_API}/versions/{dataset_id}/files"

        file_ids: dict[str, int] = dict()
        page = 1
        while True:
            if page == 1:
                url = baseurl
            else:
                url = f"{baseurl}?page={page}"

            response = self.session.get(url)
            json = response.json()
            for file_metadata in json["_embedded"]["stash:files"]:
                file_id = int(
                    file_metadata["_links"]["stash:download"]["href"].split("/")[-2]
                )
                file_name = file_metadata["path"]

                file_ids[file_name] = file_id

            if "next" in json["_links"]:
                page += 1
            else:
                break

        return file_ids

    def download(self):
        dryad_dataset_id = self.get_dryad_dataset_id()
        file_ids = self.get_dryad_file_ids(dryad_dataset_id)

        files_to_download = self.get_files_to_download()
        n_files = len(files_to_download)

        msg = f"Downloading the following {n_files} file{'s' if n_files > 1 else ''} to {self.outdir}"
        print(msg)
        for file in files_to_download:
            print(f"\t{file}")

        BLOCK_SIZE = 1024
        self.outdir.mkdir(parents=True, exist_ok=True)
        for idx, file in enumerate(files_to_download):
            file_id = file_ids[file]
            url = f"{self.DRYAD_REST_API}/files/{file_id}/download"

            output = self.outdir.joinpath(file)

            response = self.session.get(url, stream=True)
            file_size = int(response.headers.get("content-length", 0))

            pbar_desc = f"[{idx}/{n_files}] {file}"

            with (
                output.open("wb") as fdst,
                tqdm(
                    desc=pbar_desc, total=file_size, unit="B", unit_scale=True
                ) as pbar,
            ):
                for chunk in response.iter_content(BLOCK_SIZE):
                    fdst.write(chunk)
                    pbar.update(len(chunk))

        print(f"[{n_files}/{n_files}] Download finished.")

        print("Decompressing all tarballs, zip files, and gzipped files.")
        self._cleanup(files_to_download, add_outdir=True, delete_original=True)

    def gunzip(self, file: Path, delete_original: bool = True):
        parentdir = file.parent
        output = parentdir.joinpath(file.stem)
        with gzip.open(file, "rb") as fsrc, output.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)

        if delete_original:
            file.unlink()

    def decompress_directory(
        self,
        file: Path,
        tarred: bool = False,
        zipped: bool = False,
        delete_original: bool = True,
    ):
        parentdir = file.parent
        if (not tarred and not zipped) or (tarred and zipped):
            raise ValueError(
                "File must be either tarred (.tar.gz) OR zipped (.zip). Not neither and not both."
            )
        elif tarred:
            openfn = partial(tarfile.open, mode="r:gz")
            basename = file.name.rsplit(".", 2)[0]
        elif zipped:
            openfn = zipfile.ZipFile
            basename = file.stem

        with openfn(file) as fobj:
            fobj.extractall(path=self.outdir)

        # need to check if any inner files are compressed
        output = parentdir.joinpath(basename)
        inner_files = list(output.rglob("*"))
        self._cleanup(inner_files, add_outdir=False)

        if delete_original:
            file.unlink()

    def _cleanup(
        self,
        downloaded_files: list[str] | list[Path],
        add_outdir: bool = False,
        delete_original: bool = True,
    ):
        for file in downloaded_files:
            if add_outdir:
                path = self.outdir.joinpath(file)
            else:
                path = Path(file)

            if path.name.endswith(".tar.gz"):
                self.decompress_directory(
                    path, tarred=True, delete_original=delete_original
                )
            elif path.name.endswith(".gz"):
                self.gunzip(path, delete_original=delete_original)
            elif path.name.endswith(".zip"):
                self.decompress_directory(
                    path, zipped=True, delete_original=delete_original
                )

        for file in self.outdir.glob("*"):
            if file.name == "__MACOSX":
                shutil.rmtree(file)
