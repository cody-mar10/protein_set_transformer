from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ManuscriptDataArgs(BaseModel):
    aai: bool = Field(
        False,
        description="download intermediate files for AAI calculations in the manuscript (aai.tar.gz)",
    )
    fasta: bool = Field(
        False,
        description="download protein fasta files for training and test viruses (fasta.tar.gz)",
    )
    host_prediction: bool = Field(
        False,
        description="download all data associated with the host prediction proof of concept (host_prediction.tar.gz)",
    )
    readme: bool = Field(True, description="download the DRYAD README (README.md)")
    supplementary_data: bool = Field(
        False,
        description="download supplementary data directly used to make the figures in the manuscript (supplementary_data.tar.gz)",
    )
    supplementary_tables: bool = Field(
        False,
        description="download supplementary tables (supplementary_tables.zip)",
    )


class ClusterArgs(BaseModel):
    genome_clusters: bool = Field(
        False, description="download genome cluster labels (genome_clusters.tar.gz)"
    )
    protein_clusters: bool = Field(
        False, description="download protein cluster labels (protein_clusters.tar.gz)"
    )


class ModelArgs(BaseModel):
    trained_models: bool = Field(
        False, description="download trained vPST models (trained_models.tar.gz)"
    )


class EmbeddingsArgs(BaseModel):
    esm_large: bool = Field(
        False,
        description="download ESM2 large [t33_150M] PROTEIN embeddings for training and test viruses (esm-large_protein_embeddings.tar.gz)",
    )
    esm_small: bool = Field(
        False,
        description="download ESM2 small [t6_8M] PROTEIN embeddings for training and test viruses (esm-small_protein_embeddings.tar.gz)",
    )
    vpst_large: bool = Field(
        False,
        description="download vPST large PROTEIN embeddings for training and test viruses (pst-large_protein_embeddings.tar.gz)",
    )
    vpst_small: bool = Field(
        False,
        description="download vPST small PROTEIN embeddings for training and test viruses (pst-small_protein_embeddings.tar.gz)",
    )
    genome: bool = Field(
        False,
        description="download all genome embeddings for training and test viruses (genome_embeddings.tar.gz)",
    )
    genslm: bool = Field(
        False,
        description="download GenSLM ORF embeddings (genslm_protein_embeddings.tar.gz)",
    )


class DownloadArgs(BaseModel):
    all: bool = Field(
        False,
        description="download all files from the DRYAD repository",
    )
    outdir: Path = Field(Path("pstdata"), description="output directory to save files")
    embeddings: EmbeddingsArgs
    trained_models: ModelArgs
    clusters: ClusterArgs
    manuscript_data: ManuscriptDataArgs

    @model_validator(mode="after")
    def check_at_least_one_download(self) -> "DownloadArgs":
        requested: list[bool] = list()

        for field, fieldvalue in self:
            if isinstance(fieldvalue, BaseModel):
                for subfield, subfieldvalue in fieldvalue:
                    requested.append(subfieldvalue)
            elif field == "all":
                requested.append(fieldvalue)

        if not any(requested):
            raise ValueError("At least one file must be downloaded")

        return self
