# Protein Set Transformer

This repository contains the Protein Set Transformer (PST) framework for contextualizing protein language model embeddings at genome-scale to produce genome embeddings. You can use this code to train your own models. Using our foundation model pre-trained on viruses (vPST), you can also generate genome embeddings for input viruses.

For more information, see our manuscript:

Protein Set Transformer: A protein-based genome language model to power high diversity viromics.  
Cody Martin, Anthony Gitter, and Karthik Anantharaman.
*bioRxiv*, 2024, doi: [10.1101/2024.07.26.605391](https://doi.org/10.1101/2024.07.26.605391).

## Installation

### Without GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n pst -c pytorch -c pyg -c conda-forge 'python<3.12' 'pytorch>=2.0' cpuonly pyg pytorch-scatter

mamba activate pst

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/protein_set_transformer.git
```

### With GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n pst -c pytorch -c nvidia -c pyg -c conda-forge 'python<3.12' 'pytorch>=2.0' pytorch-cuda=11.8 pyg torch_scatter

mamba activate pst

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/protein_set_transformer.git
```

### Test run

Upon successful installation, you will have the `pst` executable to train, tune, and predict.

You can use the test data for a test prediction run:

```bash
pst predict \
    --file test/test_data.graphfmt.h5 \
    --accelerator cpu \
    --checkpoint test/pst-small_model.ckpt \
    --outdir test_run
```

The results from the above command are available at `test/test_run/predictions.h5`.

## Data availability

All data associated with the initial training model training can be found here: [https://doi.org/10.5061/dryad.d7wm37q8w](https://doi.org/10.5061/dryad.d7wm37q8w)

Specifically at the above link, `trained_models.tar.gz` contains both sizes of the vPST foundation model, `pst-small` and `pst-large`. Each model was trained with the same input data.

The training and test data are also available in the above data repository.

Here is a summary of each model:
| Model       | # Encoder layers | # Attention heads | # Params | Embedding dim |
| :---------- | :--------------- | :---------------- | :------- | :------------ |
| `pst-small` | 5                | 4                 | 5.4M     | 400           |
| `pst-large` | 20               | 32                | 177.9M   | 1280          |

## Embedding new genomes with the pretrained models

### 1. ESM2 protein embeddings

You will first need to generate ESM2 protein embeddings.

The source repository for ESM can be found here: [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm). At the beginning of this project, there was not a user friendly way provided by the ESM team to get protein embeddings from a protein FASTA file, so we have provided a repository to do this: [https://github.com/cody-mar10/esm_embed](https://github.com/cody-mar10/esm_embed). We plan to integrate this into the single `pst` executable for simpler usage, including creating an end-to-end pipeline to go from protein FASTA files to PST outputs.

The ESM team has now provided `esm-extract` utility to do this, but we have not yet integrated protein embeddings generated from this route.

Here is what ESM2 models are used for each vPST model:
| vPST         | ESM2                  |
| :---------- | :-------------------- |
| `pst-small` | `esm2_t30_150M_UR50D` |
| `pst-large` | `esm2_t6_8M_UR50D`    |

#### FASTA File requirements

The `esm_embed` tool we provide produces protein language model embeddings for each protein in an input FASTA file **IN THE SAME ORDER** as the sequences in the file. We plan to integrate this tool into the `pst` executable.

Thus, the following are **required** of the input FASTA file:

1. The file must be sorted to group all proteins from the same genome together
2. For the block of proteins from each genome, the proteins must be in order of their appearance in the genome.
3. The FASTA headers must look like this: `scaffold_#`, where `scaffold` is the genome scaffold name and `#` is the protein numerical ID relative to each scaffold.
    - In the event that you have multi-scaffold viruses (vMAGs, etc.), you can either manually orient the scaffolds and renumber the proteins to contiguously count from the first scaffold to the last. This is what was done with the test dataset in the manuscript.
        - We provided a utility script `pst graphify` to do this if an input mapping from scaffolds to genomes is provided. See next section.
    - TODO: We will explore a more native solution for multi-scaffold viruses that does not require an arbitrary arrangement of scaffolds that should not require changes to the model.

### 2. Convert protein embeddings to graph format

Use the `pst graphify` command to convert the ESM2 protein embeddings into graph format. You will need to protein FASTA file used to generate the embeddings, since the embeddings should be in the same order as the FASTA file. The FASTA file should be in prodigal format:
`>scaffold_ptnid # start # stop # strand ....`

If you did not keep the extra metadata on the headers, you can alternatively provide a simple tab-delimited mapping file that maps each protein name to its strand (-1 or 1 only).

Further, if you have multi-scaffold viruses, you can provide a tab-delimited file that maps the scaffold name to the genome name to count all proteins from the entire genome instead of each scaffold.

### 3. Use PST for genome embeddings and contextualized protein embeddings

Use the `pst predict` command with the input graph-formatted protein embeddings and trained model checkpoint. The test run above shows the minimum flags needed. You can also use `pst predict -h` to see what options are available, but the most important ones will be:

| Argument        | Description                                                                           |
| :-------------- | :------------------------------------------------------------------------------------ |
| `--file`        | Input graph-formatted .h5 file                                                        |
| `--outdir`      | Output directory name                                                                 |
| `--accelerator` | Device accelerator. Defaults to "gpu", so you may need to change this to "cpu"        |
| `--devices`     | Either the number of GPUs or the number of CPU threads (depending on `--accelerator`) |
| `--checkpoint`  | Which trained model checkpoint to use. See data availability above.                   |

## Manuscript

We have provided code for all analyses associated with the manuscript in the [manuscript](manuscript) folder. The README in that folder links each method section from the manuscript to a specific Jupyter notebook code implementation.

### Associated repositories

There are several other repositories associated with the model code and the manuscript:

| Repository | Description |
| :--------- | :---------- |
| [lightning-crossval](https://github.com/cody-mar10/lightning-crossval) | Our fold-synchronized cross validation strategy implemented with Lightning Fabric |
| [esm_embed](https://github.com/cody-mar10/esm_embed) | Our user-friendly way of embedding proteins from a FASTA file with ESM2 models |
| [genslm_embed](https://github.com/cody-mar10/genslm_embed) | Code to generate [GenSLM](https://github.com/ramanathanlab/genslm) ORF and genome embeddings |
| [hyena-dna-embed](https://github.com/cody-mar10/hyena-dna-embed) | Code to generate [Hyena-DNA](https://github.com/HazyResearch/hyena-dna) genome embeddings |
| [PST_host_prediction](https://github.com/cody-mar10/PST_host_prediction) | Model and evaluation code for our host prediction proof of concept analysis |
