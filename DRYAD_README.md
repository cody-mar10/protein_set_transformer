# Data from: Protein Set Transformer: A protein-based genome language model to power high diversity viromics

These datasets are associated with the manuscript "Protein Set Transformer: A protein-based genome language model powers high diversity viromics". ([https://doi.org/10.1101/2024.07.26.605391](https://doi.org/10.1101/2024.07.26.605391))

## Dataset descriptions

This statement is relevant for the file references in the manuscript code: We refer to the processed data directly used for figure making as "Supplementary **Data**". Major datasets that are used throughout the manuscript are referred to as "**Datasets**". Due to the size of the "datasets", we have provided each dataset folder as individual tarballs.

### Main data

| File | Description |
| :-- | :-- |
| `fasta.tar.gz` | Protein and gene FASTA files for the training and test datasets for the the genomes used for training the viral PST (vPST). This also includes viral genomes detected from MGnify soil metagenomes. |
| `esm_embeddings.tar.gz` | `ESM` protein embeddings for all proteins from each dataset in the graph format required as input to PST |
| `PST-TL_genome_embeddings.tar.gz` | Genome embeddings for all genomes in each train/test dataset using PST-TL models (`PST-TL-P__large`, etc) |
| `other_genome_embeddings.tar.gz` | All other genome embeddings for all genomes in each train/test dataset using PST-TL models (`PST-MLM`, `ESM`, etc) |
| `PST_training_set_PST-TL-P__small_protein_embeddings.h5` | `PST_TL-P__small` protein embeddings for PST training set proteins |
| `PST_training_set_PST-TL-P__large_protein_embeddings.h5` | `PST_TL-P__large` protein embeddings for PST training set proteins |
| `PST_training_set_PST-TL-T__small_protein_embeddings.h5` | `PST_TL-T__small` protein embeddings for PST training set proteins |
| `PST_training_set_PST-TL-T__large_protein_embeddings.h5` | `PST_TL-T__large` protein embeddings for PST training set proteins |
| `IMGVRv4_test_set_PST-TL-P__small_protein_embeddings.h5` | `PST_TL-P__small` protein embeddings for IMG/VR v4 test set proteins |
| `IMGVRv4_test_set_PST-TL-P__large_protein_embeddings.h5` | `PST_TL-P__large` protein embeddings for IMG/VR v4 test set proteins |
| `IMGVRv4_test_set_PST-TL-T__small_protein_embeddings.h5` | `PST_TL-T__small` protein embeddings for IMG/VR v4 test set proteins |
| `IMGVRv4_test_set_PST-TL-T__large_protein_embeddings.h5` | `PST_TL-T__large` protein embeddings for IMG/VR v4 test set proteins |
| `MGnify_test_set_PST-TL-P__small_protein_embeddings.h5` | `PST_TL-P__small` protein embeddings for MGnify test set proteins |
| `MGnify_test_set_PST-TL-P__large_protein_embeddings.h5` | `PST_TL-P__large` protein embeddings for MGnify test set proteins |
| `MGnify_test_set_PST-TL-T__small_protein_embeddings.h5` | `PST_TL-T__small` protein embeddings for MGnify test set proteins |
| `MGnify_test_set_PST-TL-T__large_protein_embeddings.h5` | `PST_TL-T__large` protein embeddings for MGnify test set proteins |
| `genslm_ORF_embeddings.h5` | `GenSLM` ORF embeddings for all genes from each dataset |

For genome embeddings, the order of genomes is described in supplementary table 1. Likewise, the order of proteins for protein/ORF embeddings is found in supplementary table 2.

### Model checkpoints

| File | Description |
| :-- | :-- |
| `PST-TL-P__small.ckpt.gz` | PST model trained with triplet loss and `ESM_small` input embeddings and tuned with viral protein diversity groups |
| `PST-TL-P__large.ckpt.gz` | PST model trained with triplet loss and `ESM_large` input embeddings and tuned with viral protein diversity groups |
| `PST-TL-T__small.ckpt.gz` | PST model trained with triplet loss and `ESM_small` input embeddings and tuned with viral taxonomy groups |
| `PST-TL-T__large.ckpt.gz` | PST model trained with triplet loss and `ESM_large` input embeddings and tuned with viral taxonomy groups |
| `PST-MLM.tar.gz` | PST models trained with masked language modeling. Includes both `small` and `large` models and those tuned with either viral protein diversity or viral taxonomy groups |

### Misc

| File | Description |
| :-- | :-- |
| `foldseek_databases.tar.gz` | Foldseek 3Di structure databases for both test datasets |
| `supplementary_tables.zip`| Supplementary tables associated with the manuscript |
| `supplementary_data.tar.gz` | Supplementary data used for figure making, processed from the rest the dataset files here |
| `host_prediction.tar.gz` | Data associated with the host prediction proof-of-concept analyses, including the protein FASTA files for both the viruses and hosts, knowledge graphs for each node embedding type, and the best trained models |
| `protein_clusters.tar.gz` | Embedding-based protein clusters for test datasets |
| `genome_clusters.tar.gz` | Embedding-based genome clusters for test datasets |

## Directory structure

After downloading and untarring/unzipping directories, your directory where you've downloaded these files should look like this:

```text
.
|__fasta/
|  |__IMGVRv4_test_set.faa.gz # protein FASTA file for vPST IMG/VR v4 test viruses
|  |__IMGVRv4_test_set.ffn.gz # nucleotide ORF FASTA file for vPST IMG/VR v4 test viruses
|  |__MGnify_test_set.faa.gz  # protein FASTA file for vPST MGnify test viruses
|  |__MGnify_test_set.ffn.gz  # nucleotide ORF FASTA file for vPST MGnify test viruses
|  |__MGnify_test_set.fna.gz  # genomes FASTA file for vPST MGnify test viruses
|  |__training_set.faa.gz     # protein FASTA file for vPST training viruses
|  |__training_set.ffn.gz     # nucleotide ORF FASTA file for vPST training viruses
|
|__esm_embeddings/
|  |__IMGVRv4_test_set_ESM__large.graphfmt.h5 # ESM__large embeddings for IMGVRv4 test viruses in graph format for PST input
|  |__IMGVRv4_test_set_ESM__small.graphfmt.h5 # ESM__small embeddings for IMGVRv4 test viruses in graph format for PST input
|  |__MGnify_test_set_ESM__large.graphfmt.h5  # ESM__large embeddings for MGnify test viruses in graph format for PST input
|  |__MGnify_test_set_ESM__small.graphfmt.h5  # ESM__small embeddings for MGnify test viruses in graph format for PST input
|  |__PST_training_set_ESM__large.graphfmt.h5 # ESM__large embeddings for PST training viruses in graph format for PST input
|  |__PST_training_set_ESM__small.graphfmt.h5 # ESM__small embeddings for PST training viruses in graph format for PST input
|
|__PST-TL_genome_embeddings/
|  |__PST_training_set_PST-TL_genome_embeddings.h5 # PST-TL genome embedding matrices for training viruses
|  |__IMGVRv4_test_set_PST-TL_genome_embeddings.h5 # PST-TL genome embedding matrices for IMGVRv4 test viruses
|  |__MGnify_test_set_PST-TL_genome_embeddings.h5  # PST-TL genome embedding matrices for MGnify test viruses
|
|__other_genome_embeddings/
|  |__PST_training_set_other_genome_embeddings.h5 # all other genome embedding matrices for training viruses
|  |__IMGVRv4_test_set_other_genome_embeddings.h5 # all other genome embedding matrices for IMGVRv4 test viruses
|  |__MGnify_test_set_other_genome_embeddings.h5  # all other genome embedding matrices for MGnify test viruses
|
|__genome_clusters/
|  |__IMGVRv4_test_set_genome_clusters_ang_sim_k15_res0.9.h5 # embedding-based cluster assignments for IMGVRv4 test viruses
|  |__MGnify_test_set_genome_clusters_k15_res0.9.h5          # embedding-based cluster assignments for MGnify test viruses
|
|__protein_clusters/
|  |__IMGVRv4_test_set_protein_clusters_ang_sim_k15_res0.9.h5 # embedding-based cluster assignments for IMGVRv4 test viruses
|  |__MGnify_test_set_protein_clusters_k15_res0.9.h5          # embedding-based cluster assignments for MGnify test viruses
|
|__host_prediction/
|  |__fasta/
|  |  |__all_hosts.faa.gz               # protein FASTA file for all host proteins
|  |  |__host_scaffold_to_genome.tsv.gz # mapping file to link host genomes with each host scaffold
|  |  |__virus_test_set.faa.gz          # protein FASTA file for iPHoP test viruses
|  |  |__virus_training_set.faa.gz      # protein FASTA file for iPHoP training viruses
|  |
|  |__knowledge_graphs/
|  |  |__cherry_knowledge_graph.pt    # knowledge graph with tetra nucleotide frequency node embeddings - identical with kmer
|  |  |__kmer_knowledge_graph.pt      # knowledge graph with tetra nucleotide frequency node embeddings - identical with cherry
|  |  |__ESM_knowledge_graph.pt       # knowledge graph with ESM__large (ESM2_t30_150M) node embeddings
|  |  |__PST-TL-T_knowledge_graph.pt  # knowledge graph with PST-TL-T__large node embeddings
|  |  |__PST-TL-P_knowledge_graph.pt  # knowledge graph with PST-TL-P__large node embeddings
|  |  |__PST-MLM-T_knowledge_graph.pt # knowledge graph with PST-MLM-T__large node embeddings
|  |  |__PST-MLM-P_knowledge_graph.pt # knowledge graph with PST-MLM-P__large node embeddings
|  |
|  |__trained_models/
|     |__CHERRY.ckpt    # best model with CHERRY-specific hyperparams
|     |__kmer.ckpt      # best model trained with tetranucleotide (kmer) node embeddings
|     |__ESM.ckpt       # best model trained with ESM__large (ESM2_t30_150M) node embeddings
|     |__PST-TL-T.ckpt  # best model trained with PST-TL-T__large node embeddings
|     |__PST-TL-P.ckpt  # best model trained with PST-TL-P__large node embeddings
|     |__PST-MLM-T.ckpt # best model trained with PST-MLM-T__large node embeddings
|     |__PST-MLM-P.ckpt # best model trained with PST-MLM-P__large node embeddings
|
|__foldseek_databases/
   |__IMGVRv4_test_set/                  # foldseek database for IMGVRv4 test virus proteins
   |  |__IMGVRv4_test_set_fs_db
   |  |__IMGVRv4_test_set_fs_db.dbtype
   |  |__IMGVRv4_test_set_fs_db.index
   |  |__IMGVRv4_test_set_fs_db.lookup
   |  |__IMGVRv4_test_set_fs_db.source
   |  |__IMGVRv4_test_set_fs_db_h
   |  |__IMGVRv4_test_set_fs_db_h.dbtype
   |  |__IMGVRv4_test_set_fs_db_h.index
   |
   |__MGnify_test_set/                   # foldseek database for MGnify test virus proteins
   |__MGnify_test_set_fs_db
   |__MGnify_test_set_fs_db.dbtype
   |__MGnify_test_set_fs_db.index
   |__MGnify_test_set_fs_db.lookup
   |__MGnify_test_set_fs_db.source
   |__MGnify_test_set_fs_db_h
   |__MGnify_test_set_fs_db_h.dbtype
   |__MGnify_test_set_fs_db_h.index

```

## Supplementary data file structures

The supplementary data are organized by figure in the manuscript. We indicate with figure panel each file was used for:

```text
.
├── fig2
│  ├── ASI_vs_embedding_sim_pearson_correlation.tsv
│  ├── IMGVR_distant_genomes_ASI_vs_embedding_sim.tsv
│  ├── IMGVR_similar_genomes_ASI_vs_embedding_sim.tsv
│  ├── MGnify_distant_genomes_ASI_vs_embedding_sim.tsv
│  └── MGnify_similar_genomes_ASI_vs_embedding_sim.tsv
├── fig3
│  ├── attention_scores.tsv
│  ├── functional_module_detection.tsv
│  └── PHROG_functional_co-clustering_enrichment.tsv
├── fig4
│  ├── annotation_transfer.tsv
│  └── capsid_detection_rates.tsv
├── fig5
│  └── iPHoP_test_set_prediction_summary.tsv
└── supp_figs
   ├── supp_fig3_dataset_aai_comparisons.tsv
   ├── supp_fig24_iphop_datasets_aai_comparisons.tsv
   ├── supp_fig26_connected_component_average_dice_scores.h5
   ├── supp_fig26_genome_mapped_to_protein_diversity_connected_component.tsv
   └── supp_fig26_protein_diversity_dice_graph.csr.h5
```

## Supplementary table descriptions

| Table Number | Description |
| :-- | :-- |
| 1 | Full metadata associated with the 103,589 train, 151,255 IMGVRv4 test, and ___ MGnify test vPST viruses, including scaffold names, assigned taxonomy, and host taxonomy (if any), and the genome ID we used that correspond to the order in many `.h5` files. |
| 2 | Protein metadata that links each protein to a corresponding genome from `Supplementary Table 1`. VOG and PHROG annotations (if any) for each protein are included. |
| 3 | PST hyperparameters sampled for tuning |
| 4 | Final PST hyperparameters chosen after model tuning with LeaveOneGroupOut cross validations |
| 5 | Reproduction of the PHROG profile descriptions that include our function re-categorizations |
| 6 | Functional categorization of each VOG HMM, along with the HMM description |
| 7 | Regex patterns used to categorize each VOG HMM based on its annotation description |
| 8 | Host prediction genome metadata for the 3628 train and 1636 test viruses. This includes the specific host genomes for each viral genomes. |
| 9 | Hyperparameters sampled for the graph-based host predictions models we trained. |
| 10 | Hyperparameter tuning trial summary for training PST models |
| 11 | Summary of pretrained PST models, including number of trained epochs, batch accumulation size, training time, and number of parameters. |

## H5 file structure

The embedding files were compressed using `blosc2:lz4hc`, so you may need to specifically use [pytables](https://www.pytables.org) (which is used in the PST code) to open these files if your H5-library does not come with these compressors.

### Genome embeddings

The structure of each genome embedding H5 file is:

```text
*_genome_embeddings.h5
|__GENOME-EMBEDDING
```

Where each field contains the genome embedding matrix `Shape: (Number of viruses, Embedding dimension)` for the viruses in that dataset.

We have split up genome embeddings into separate datasets so that the most relevant embeddings (`PST-TL`) can be downloaded more easily. Specifically, the split is as follows:

1. PST-TL genome embeddings
   1. `PST-TL-P__large` - 800 embed dim
   2. `PST-TL-P__small` - 400 embed dim
   3. `PST-TL-T__large` - 1280 embed dim
   4. `PST-TL-T__small` - 400 embed dim
2. Other genome embeddings
   1. `ESM__large`          - 640 embed dim
   2. `ESM__small`          - 320 embed dim
   3. `PST-CTX-TL-P__large` - 800 embed dim
   4. `PST-CTX-TL-P__small` - 400 embed dim
   5. `PST-CTX-TL-T__large` - 1280 embed dim
   6. `PST-CTX-TL-T__small` - 400 embed dim
   7. `PST-MLM-P__large`    - 1920 embed dim
   8. `PST-MLM-P__small`    - 960 embed dim
   9. `PST-MLM-T__large`    - 1920 embed dim
   10. `PST-MLM-T__small`   - 960 embed dim
   11. `genslm`             - 512 embed dim
   12. `hyena-dna`          - 256 embed dim
   13. `kmer`               - 256 embed dim

### Protein embeddings

#### ESM2 protein embeddings

The minimum structure of each graph-formatted ESM protein embedding H5 file is:

```text
esm_embeddings/*ESM*.graphfmt.h5
|__data     # (N proteins, Embed dim) stacked ESM2 protein embeddings
|__ptr      # (N genomes + 1,) index pointer to access protein embeddings for each scaffold
|__sizes    # (N genomes,) number of proteins encoded by each scaffold
|__strand   # (N proteins,) strand label {-1, 1} for each protein according to pyrodigal/prodigal-gv
```

Note: A scaffold is a contiguous nucleotide sequence, but there is support for multi-scaffold genomes, like MAGs/vMAGs. For example, the `IMG/VR v4` dataset additionally has a `genome_label` field since there are multi-scaffold viruses in that dataset, which assigns each scaffold to a genome.

The `data` field is sorted by each genome and the order of proteins encoded in each genome. The protein embeddings are stored in CSR format, so the `ptr` field is an index pointer to compute offsets and start/stops for each genome. This format is used by [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#mini-batches).

**These files are used as inputs to PST models.**

#### PST-TL protein embeddings

The PST-TL protein embeddings are all separate files since they are large. They all have a single field that is the same name as the embedding type, ie:

`PST_training_set_PST-TL-T__small_protein_embeddings.h5` has a single field called `PST-TL-T__small` that contains the `PST-TL-T__small` protein embeddings for the PST training set.

To prevent data duplication, we did not provide the genome pointers used for accessing chunks from the above matrices that correspond to specific genomes. These can be accessed from the ESM2 graph-formatted `.h5` files.

Note: Due to the size of the `PST-MLM` models, the precomputed protein embeddings cannot be easily provided, as this would require >300GB just for these embeddings alone.

#### GenSLM ORF embeddings

```text
genslm_orf_embeddings.h5
|__IMGVRv4 # (N proteins, Embed dim) stacked orf embeddings for all IMGVRv4 test virus proteins
|__MGnify  # (N proteins, Embed dim) stacked orf embeddings for all MGnify test virus proteins
|__PST     # (N proteins, Embed dim) stacked orf embeddings for all train virus proteins
```

To prevent data duplication, we did not provide the genome pointers used for accessing chunks from the above matrices that correspond to specific genomes. These can be accessed from the ESM2 graph-formatted `.h5` files.

## Code/Software

Code to use this data and reproduce manuscript analyses can be found at: [https://github.com/AnantharamanLab/protein_set_transformer](https://github.com/AnantharamanLab/protein_set_transformer).
