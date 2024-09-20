# Data from: Protein Set Transformer: A protein-based genome language model to power high diversity viromics

These datasets are associated with the manuscript "Protein Set Transformer: A protein-based genome language model powers high diversity viromics". ([https://doi.org/10.1101/2024.07.26.605391](https://doi.org/10.1101/2024.07.26.605391))

## Dataset descriptions

This statement is relevant for the file references in the manuscript code: We refer to the processed data directly used for figure making as "Supplementary **Data**". Major datasets that are used throughout the manuscript are referred to as "**Datasets**". Due to the size of the "datasets", we have provided each dataset folder as individual tarballs.

| Tarred file                           | Description                                                                                                                                                                                                                                                                                                                                                 |
| :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fasta.tar.gz`                        | Protein and gene FASTA files for the training and test datasets for the the genomes used for training the viral PST (vPST)                                                                                                                                                                                                                                  |
| `genome_embeddings.tar.gz`            | Genome embeddings for the genomes in the training and test vPST datasets. The genome embeddings include ESM2-genome-averaged (`esm-small`, `esm-large`), PST contextualized protein genome-averaged (`ctx-avg-small`, `ctx-avg-large`), PST (`pst-small`, `pst-large`), tetra nucleotide frequency (`kmer`), GenSLM (`genslm`), and Hyena-DNA (`hyena-dna`) |
| `pst-large_protein_embeddings.tar.gz` | vPST (large) protein embeddings for the training and test vPST datasets                                                                                                                                                                                                                                                                                     |
| `pst-small_protein_embeddings.tar.gz` | vPST (small) protein embeddings for the training and test vPST datasets                                                                                                                                                                                                                                                                                     |
| `esm-large_protein_embeddings.tar.gz` | ESM2 large (t30\_150M) protein embeddings for the training and test vPST datasets. These are graph-formatted for direct use with the PST code                                                                                                                                                                                                               |
| `esm-small_protein_embeddings.tar.gz` | ESM2 large (t6\_8M) protein embeddings for the training and test vPST datasets. These are graph-formatted for direct use with the PST code                                                                                                                                                                                                                  |
| `genslm_protein_embeddings.tar.gz`    | GenSLM nucleotide ORF embeddings for the training and test vPST datasets                                                                                                                                                                                                                                                                                    |
| `trained_models.tar.gz`               | 2 trained vPST models: `pst-small` and `pst-large`                                                                                                                                                                                                                                                                                                          |
| `host_prediction.tar.gz`              | Data associated with the host prediction proof-of-concept analyses, including the protein FASTA files for both the viruses and hosts, knowledge graphs for each node embedding type, and the best trained models                                                                                                                                            |
| `aai.tar.gz`                          | Raw protein-protein search data and subsequent AAI summary for the vPST test viruses                                                                                                                                                                                                                                                                        |
| `protein_clusters.tar.gz`             | Sequence identity- and embedding-based protein clusters                                                                                                                                                                                                                                                                                                     |
| `genome_clusters.tar.gz`              | Embedding-based genome clusters                                                                                                                                                                                                                                                                                                                             |
| `supplementary_tables.zip`            | Supplementary tables associated with the manuscript                                                                                                                                                                                                                                                                                                         |
| `supplementary_data.tar.gz`           | Supplementary data used for figure making, processed from the rest the dataset files here                                                                                                                                                                                                                                                                   |

For genome embeddings, the order of genomes is described in supplementary table 1. Likewise, the order of proteins for protein embeddings is found in supplementary table 2.

## Dataset file structure

After downloading and untarring/unzipping the dataset files, your directory where you've downloaded these files should look like this:

```text
./
|__fasta/
|  |__test_set.faa # protein FASTA file for vPST test viruses
|  |__test_set.ffn # nucleotide ORF FASTA file for vPST test viruses
|  |__training_set.faa # protein FASTA file for vPST training viruses
|  |__training_set.ffn # nucleotide ORF FASTA file for vPST test viruses
|
|__genome_embeddings/
|  |__training_set_genome_embeddings.h5 # genome embedding matrices for vPST training viruses
|  |__test_set_genome_embeddings.h5 # genome embedding matrices for vPST test viruses
|
|__genome_clusters/
|  |__aai-based_genome_clusters.h5 # cluster assignments based on AAI-clustering
|  |__embedding-based_genome_clusters.h5 # cluster assignments based on clustering genome embeddings
|
|__protein_clusters/
|  |__sequence_identity_clusters.tsv # mmseqs2 cluster assignments
|  |__embedding-based_protein_clusters_per_genome_cluster.h5 # cluster assignments based on clustering protein embeddings within genome clusters
|
|__trained_models/
|  |__pst-small_trained_model.ckpt # small vPST model trained using ESM2_t6_8M protein embeddings with optimal hyperparams
|  |__pst-large_trained_model.ckpt # large vPST model trained using ESM2_t30_150M protein embeddings with optimal hyperparams
|
|__host_prediction/
|  |__fasta/
|  |  |__all_hosts.faa # protein FASTA file for all host proteins
|  |  |__host_scaffold_to_genome.tsv # mapping file to link host genomes with each host scaffold
|  |  |__virus_test_set.faa # protein FASTA file for iPHoP test viruses
|  |  |__virus_training_set.faa # protein FASTA file for iPHoP training viruses
|  |
|  |__knowledge_graphs/
|  |  |__cherry_knowledge_graph.pt # knowledge graph with tetra nucleotide frequency node embeddings - identical with kmer
|  |  |__kmer_knowledge_graph.pt # knowledge graph with tetra nucleotide frequency node embeddings - identical with cherry
|  |  |__esm-large_knowledge_graph.pt # knowledge graph with esm-large (ESM2_t30_150M) node embeddings
|  |  |__pst-large_knowledge_graph.pt # knowledge graph with pst-large node embeddings
|  |
|  |__trained_models/
|     |__cherry_trained_host_prediction_model.ckpt # best model with CHERRY-specific hyperparams
|     |__kmer_trained_host_prediction_model.ckpt # best model trained with tetranucleotide (kmer) node embeddings
|     |__esm-large_trained_host_prediction_model.ckpt # best model trained with esm-large (ESM2_t30_150M) node embeddings
|     |__pst-large_trained_host_prediction_model.ckpt # best model trained with pst-large node embeddings
|     |__best_model_performance.tsv # training curves for the models here
|
###### protein embeddings should be in folders named according to the protein embedding ######
|__esm-large/
|  |__test_set_esm-large_inputs.graphfmt.h5 # stacked ESM2_t30_150M protein embeddings for all vPST test viruses
|  |__training_set_esm-large_inputs.graphfmt.h5 # stacked ESM2_t30_150M protein embeddings for all vPST training viruses
|
|__esm-small/
|  |__test_set_esm-small_inputs.graphfmt.h5 # stacked ESM2_t6_8M protein embeddings for all vPST test viruses
|  |__training_set_esm-small_inputs.graphfmt.h5 # stacked ESM2_t6_8M protein embeddings for all vPST training viruses
|
|__pst-large/
|  |__test_set_pst-large_inputs.h5 # vPST large protein embeddings for all vPST test viruses
|  |__training_set_pst-large_inputs.h5 # vPST large protein embeddings for all vPST training viruses
|
|__pst-small/
|  |__test_set_pst-small_inputs.h5 # vPST small protein embeddings for all vPST test viruses
|  |__training_set_pst-small_inputs.h5 # vPST small protein embeddings for all vPST training viruses
|
|__genslm/
   |__genslm_orf_embeddings.h5 # GenSLM ORF embeddings for both vPST train and test viruses
```

In several cases, there are `README` within these directories as well, particularly for the fields present in the `.h5` files.

## Supplementary data file structures

The supplementary data are organized by figure in the manuscript. We indicate with figure panel each file was used for:

```text
.
├── ext_data_fig2
│  └── train_test_AAI_summary.tsv # panel A
│
├── ext_data_fig3
│  ├── final_models_triplet_loss.tsv # panel E
│  ├── pst-large_tuning_history.db # panels A, B, and D
│  └── pst-small_tuning_history.db # panels A-C
│
├── ext_data_fig4
│  └── embedding-based_genome_clusters_taxonomic_purity.tsv # panels A and B
│
├── ext_data_fig5
│  ├── duplodnaviria_cluster_viruses.tsv # panel C; symlinked to: -> ../fig3/duplodnaviria_cluster_viruses.tsv
│  ├── monodnaviria_cluster_viruses.tsv # panel C; symlinked to: -> ../fig3/monodnaviria_cluster_viruses.tsv
│  └── pst-large_mmseqs_cluster_attention.tsv # panel A
│
├── ext_data_fig6
│  ├── embedding-based_protein_cluster_functional_purity.tsv # panel B
│  └── embedding-based_protein_clusters_summary.tsv # panel A
│
├── ext_data_fig7
│  └── phrog_function_co-clustering.tsv entire figure; symlinked to: -> ../fig3/phrog_function_co-clustering.tsv
│
├── ext_data_fig8
│  └── function_module_proportions.tsv panels A and B; symlinked to: -> ../fig3/function_module_proportions.tsv
│
├── ext_data_fig9
│  ├── annotation_transfer_to_unannotated_proteins.tsv # panel B
│  └── detected_capsid_proportions.tsv panel A; symlinked to: -> ../fig4/detected_capsid_proportions.tsv
│
├── ext_data_fig10
│  ├── host_prediction_recall.tsv # panel D
│  ├── intermediate_model_performance.tsv # panel A
│  ├── iphop_test_set_pst_train_set_AAI_summary.tsv # panel B
│  └── iphop_train_set_AAI_summary.tsv # panel C
│
├── fig2
│  ├── aai-based_clusters_AAI_summary.tsv # panel E
│  ├── embedding-based_clusters_AAI_summary.tsv # panel E
│  └── embedding-based_clusters_summary.tsv # panels B-D
│
├── fig3
│  ├── duplodnaviria_cluster_viruses.tsv # panel B
│  ├── function_module_proportions.tsv # panel D
│  ├── monodnaviria_cluster_viruses.tsv # panel B
│  ├── phrog_function_co-clustering.tsv # panel C
│  └── pst-large_per-protein_attention.tsv # panel A
│
├── fig4
│  ├── alphafold3/ # contains alphafold3 web server outputs for 2 proteins indicated in manuscript -> used for panels A and B
│  ├── annotation_transfer_to_unannotated_proteins_slopes.tsv # panel D
│  ├── capsid_candidates_structural_search_against_pdb.tsv # raw data for panel C
│  ├── capsid_pdb_ids.txt # panel C
│  └── detected_capsid_proportions.tsv # processed data for panel C
│
└── fig5
   └── host_prediction_recall.tsv # panel B
```

## Supplementary table descriptions

| Table Number | Description                                                                                                                                                                                                                          |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1            | Full metadata associated with the 103,589 train and  151,255 test vPST viruses, including scaffold names, assigned taxonomy, and host taxonomy (if any), and the genome ID we used that correspond to the order in many `.h5` files. |
| 2            | Protein metadata that links each protein to a corresponding genome from `Supplementary Table 1`. VOG and PHROG annotations (if any) for each protein are included.                                                                   |
| 3            | PST hyperparameters sampled for tuning                                                                                                                                                                                               |
| 4            | Final PST hyperparameters chosen after model tuning with modified leave-one-group-out cross validation                                                                                                                               |
| 5            | Reproduction of the PHROG profile descriptions that include our function re-categorizations                                                                                                                                          |
| 6            | Functional categorization of each VOG HMM, along with the HMM description                                                                                                                                                            |
| 7            | Regex patterns used to categorize each VOG HMM based on its annotation description                                                                                                                                                   |
| 8            | Host prediction genome metadata for the 3628 train and 1636 test viruses. This includes the specific host genomes for each viral genomes.                                                                                            |
| 9            | Hyperparameters sampled for the graph-based host predictions models we trained.                                                                                                                                                      |

## H5 file structure

The genome embedding files were compressed using `blosc:lz4`, so you may need to specifically use [pytables](https://www.pytables.org) (which is used in the PST code) to open these files if your H5-library does not come with these compressors.

The protein embedding files were similarly compressed using `blosc2:lz4`.

### Genome embeddings

The structure of each genome embedding H5 file is:

```text
*_genome_embeddings.h5
|__ctx-avg-large
|__ctx-avg-small
|__esm-large
|__esm-small
|__genslm
|__hyena-dna
|__kmer
|__pst-large
|__pst-small
```

Where each field contains the genome embedding matrix `Shape: (Number of viruses, Embedding dimension)` for the viruses in that dataset.

### Protein embeddings

#### ESM2 protein embeddings

The structure of each protein embedding H5 file is:

```text
*_inputs.graphfmt.h5
|__class_id # (N genomes,) viral realm label for each genome
|__data     # (N proteins, Embed dim) stacked ESM2 protein embeddings
|__ptr      # (N genomes + 1,) index pointer to access protein embeddings for each genome
|__sizes    # (N genomes,) number of proteins encoded by each genome
|__strand   # (N proteins,) strand label {-1, 1} for each protein according to pyrodigal/prodigal-gv
```

The `data` field is sorted by each genome and the order of proteins encoded in each genome. The protein embeddings are stored in CSR format, so the `ptr` field is an index pointer to compute offsets and start/stops for each genome. This format is used by [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#mini-batches). The `class_id` is not required for inference.

#### vPST protein embeddings

```text
*_contextualized_proteins.h5
|__attn # (N proteins, N attn heads) per-head attention scores for each protein, normalized by each genome
|__ctx_ptn # (N proteins, Embed dim) stacked vPST contextualized protein embeddings
```

To prevent data duplication, we did not provide the genome pointers used for accessing chunks from the above matrices that correspond to specific genomes. These can be accessed from the ESM2 graph-formatted `.h5` files.

#### GenSLM ORF embeddings

```text
genslm_orf_embeddings.h5
|__training # (N proteins, Embed dim) stacked orf embeddings for all train virus proteins
|__test # (N proteins, Embed dim) stacked orf embeddings for all test virus proteins
```

To prevent data duplication, we did not provide the genome pointers used for accessing chunks from the above matrices that correspond to specific genomes. These can be accessed from the ESM2 graph-formatted `.h5` files.

## Code/Software

Code to use this data and reproduce manuscript analyses can be found at: [https://github.com/AnantharamanLab/protein_set_transformer](https://github.com/AnantharamanLab/protein_set_transformer).
