# Protein Set Transformer

This repository contains the Protein Set Transformer (PST) framework for contextualizing protein language model embedding at genome-scale to produce genome embeddings. You can use this code to train your own models. Using our foundation model pre-trained on viruses (vPST), you can also generate genome embeddings for input viruses.

For more information, see our manuscript:

Protein Set Transformer: A genome-scale protein language model powers high diversity viromics studies.
Cody Martin, Anthony Gitter, and Karthik Anantharaman.
*bioRxiv*, 2024, doi: ADD THIS.

## Installation

### Without GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n pst -c pytorch -c pyg -c conda-forge 'pytorch>=2.0' cpuonly pyg pytorch-scatter

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/protein_set_transformer.git
```

### With GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n pst -c pytorch -c nvidia -c pyg -c conda-forge 'pytorch>=2.0' pytorch-cuda=11.8 pyg torch_scatter

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

### ESM2 protein embeddings

You will first need to generate ESM2 protein embeddings.

However, the source repository for ESM can be found here: [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm). At the beginning of this project, there was not a user friendly way from the ESM team to get protein embeddings from a protein FASTA file, so we have provided a repository to do this: [https://github.com/cody-mar10/esm_embed](https://github.com/cody-mar10/esm_embed).

The ESM team has now provided `esm-extract` utility to do this, but we have not yet integrated protein embeddings generated from this route.

Here is what ESM2 models are used for each vPST model:
| PST         | ESM2                  |
| :---------- | :-------------------- |
| `pst-small` | `esm2_t30_150M_UR50D` |
| `pst-large` | `esm2_t6_8M_UR50D`    |

#### FASTA File requirements

The `esm_embed` tool we provide produces protein language model embeddings for each protein in an input FASTA file **IN THE SAME ORDER** as the sequences in the file.

Thus, the following are **required** of the input FASTA file:

1. The file must be sorted to group all proteins from the same genome together
2. For the block of proteins from each genome, the proteins must be in order of their appareance in the genome.
3. The FASTA headers must look like this: `scaffold_#`, where `scaffold` is the genome scaffold name and `#` is the protein numerical ID relative to each scaffold.
    - In the event that you have multi-scaffold viruses (vMAGs, etc.), you can either manually orient the scaffolds and renumber the proteins to contiguously count from the first scaffold to the last. This is what was done with the test dataset in the manuscript.
        - TODO: we will provide a utility script to do this if an input mapping from scaffolds to genomes is provided.
    - TODO: We will explore a more native solution for multi-scaffold viruses that does not require an arbitrary arrangement of scaffolds.

### Convert protein embeddings to graph format

TODO: need to add a utility script for this

### Use PST for genome embeddings and contextualized protein embeddings

Use the `pst predict` command with the input graph-formatted protein embeddings and trained model checkpoint. The test run above shows the minimum flags needed. You can also use `pst predict -h` to see what options are available.

## Manuscript

- Talk about `manuscript/` folder
- Talk about submodules that are associated with manuscript analyses
