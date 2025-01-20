from attrs import define


@define
class ManuscriptDataArgs:
    """MANUSCRIPT DATA"""

    aai: bool = False
    """download AAI data for training and test viruses (aai.tar.gz)"""

    fasta: bool = False
    """download protein fasta files for training and test viruses (fasta.tar.gz)"""

    host_prediction: bool = False
    """download all data associated with the host prediction proof of concept 
    (host_prediction.tar.gz)"""

    readme: bool = True
    """download the DRYAD README (README.md)"""

    supplementary_data: bool = False
    """download supplementary data directly used to make the figures in the manuscript 
    (supplementary_data.tar.gz)"""

    supplementary_tables: bool = False
    """download supplementary tables (supplementary_tables.zip)"""


@define
class ClusterArgs:
    """CLUSTER DATA"""

    genome_clusters: bool = False
    """download genome cluster labels (genome_clusters.tar.gz)"""

    protein_clusters: bool = False
    """download protein cluster labels (protein_clusters.tar.gz)"""


@define
class ModelArgs:
    """PRETRAINED MODELS"""

    trained_models: bool = False
    "download trained vPST models (trained_models.tar.gz)"


@define
class EmbeddingsArgs:
    """EMBEDDINGS"""

    esm_large: bool = False
    """download ESM2 large [t33_150M] PROTEIN embeddings for training and test viruses 
    (esm-large_protein_embeddings.tar.gz)"""

    esm_small: bool = False
    """download ESM2 small [t6_8M] PROTEIN embeddings for training and test viruses 
    (esm-small_protein_embeddings.tar.gz)"""

    vpst_large: bool = False
    """download vPST large PROTEIN embeddings for training and test viruses 
    (pst-large_protein_embeddings.tar.gz)"""

    vpst_small: bool = False
    """download vPST small PROTEIN embeddings for training and test viruses 
    (pst-small_protein_embeddings.tar.gz)"""

    genome: bool = False
    """download all genome embeddings for training and test viruses (genome_embeddings.tar.gz)"""

    genslm: bool = False
    """download GenSLM ORF embeddings (genslm_protein_embeddings.tar.gz)"""
