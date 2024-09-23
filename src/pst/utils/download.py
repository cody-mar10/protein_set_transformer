from __future__ import annotations

import gzip
import re
import shutil
import tarfile
import zipfile
from functools import partial
from pathlib import Path

import requests
from pydantic import BaseModel
from tqdm import tqdm

from pst.utils.cli.download import DownloadArgs


class DryadDownloader:
    # dataset doi
    DRYAD_DOI_REST_API = r"doi%3A10.5061%2Fdryad.d7wm37q8w"
    DRYAD_REST_API = "https://datadryad.org/api/v2"

    def __init__(self, args: DownloadArgs):
        self.args = args
        self.filename_map = self.field2filename()

        self._session: requests.Session | None = None

    def field2filename(self) -> dict[str, str]:
        filenames: dict[str, str] = dict()
        file_pattern = re.compile(r"\((.*)\)$")

        for fieldname, fieldvalue in self.args:
            if isinstance(fieldvalue, BaseModel):
                for subfieldname, info in fieldvalue.model_fields.items():
                    desc: str = info.description  # type: ignore
                    filenames[subfieldname] = file_pattern.findall(desc)[0]

        return filenames

    def get_files_to_download(self) -> list[str]:
        files: list[str] = list()

        download_all = self.args.all

        for field, fieldvalue in self.args:
            if isinstance(fieldvalue, BaseModel):
                for subfield, subfieldvalue in fieldvalue:
                    if subfieldvalue or download_all:
                        files.append(self.filename_map[subfield])

        return files

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()

        return self._session

    def get_dryad_dataset_id(self) -> int:
        url = f"{self.DRYAD_REST_API}/datasets/{self.DRYAD_DOI_REST_API}"

        response = self.session.get(url)
        dryad_dataset_id = int(
            response.json()["_links"]["stash:version"]["href"].split("/")[-1]
        )

        return dryad_dataset_id

    def get_dryad_file_ids(self, dataset_id: int) -> dict[str, int]:
        url = f"{self.DRYAD_REST_API}/versions/{dataset_id}/files"

        response = self.session.get(url)
        file_ids: dict[str, int] = dict()

        for file_metadata in response.json()["_embedded"]["stash:files"]:
            file_id = int(
                file_metadata["_links"]["stash:download"]["href"].split("/")[-2]
            )
            file_name = file_metadata["path"]

            file_ids[file_name] = file_id

        return file_ids

    def download(self):
        dryad_dataset_id = self.get_dryad_dataset_id()
        file_ids = self.get_dryad_file_ids(dryad_dataset_id)
        files_to_download = self.get_files_to_download()
        n_files = len(files_to_download)

        BLOCK_SIZE = 1024
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        print(
            f"Downloading {n_files} file{'s' if n_files > 1 else ''} to {self.args.outdir}"
        )

        for idx, file in enumerate(files_to_download):
            file_id = file_ids[file]
            url = f"{self.DRYAD_REST_API}/files/{file_id}/download"

            output = self.args.outdir.joinpath(file)

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
            fobj.extractall(path=self.args.outdir)

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
                path = self.args.outdir.joinpath(file)
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

        for file in self.args.outdir.glob("*"):
            if file.name == "__MACOSX":
                shutil.rmtree(file)


def download(args: DownloadArgs):
    downloader = DryadDownloader(args)
    downloader.download()
