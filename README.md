# Protein Set Transformer

This repository contains the Protein Set Transformer (PST) framework for contextualizing protein language model embeddings at genome-scale to produce genome embeddings. You can use this code to train your own models. Using our foundation model pre-trained on viruses (vPST), you can also generate genome embeddings for input viruses.

For more information, see our manuscript:

Protein Set Transformer: A protein-based genome language model to power high diversity viromics.  
Cody Martin, Anthony Gitter, and Karthik Anantharaman.
*bioRxiv*, 2024, doi: [10.1101/2024.07.26.605391](https://doi.org/10.1101/2024.07.26.605391).

## Installation

You can try simply doing:

```bash
pip install ptn-set-transformer
```

But I prefer to manually setup the PyTorch installation to control CPU/GPU availability.

This full installation can be achieved with `mamba` and `pip`, which should take no more than 5 minutes.

Note: you will likely need to link your git command line interface with an online github account. Follow [this link](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git#setting-up-git) for help setting up git at the command line.

### Without GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n pst -c pytorch -c pyg -c conda-forge 'python<3.12' 'pytorch>=2.0' cpuonly pyg pytorch-scatter

mamba activate pst
pip install ptn-set-transformer
```

### With GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n pst -c pytorch -c nvidia -c pyg -c conda-forge 'python<3.12' 'pytorch>=2.0' pytorch-cuda=11.8 pyg pytorch-scatter

mamba activate pst
pip install ptn-set-transformer
```

### Installing for training a new PST

We implemented a hyperparameter tuning cross validation workflow implemented using [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) in a base library called [lightning-crossval](https://github.com/cody-mar10/lightning-crossval). Part of our specific implementation for hyperparameter tuning is also implemented in the PST library.

If you want to include the optional dependendings for training a new PST, you can follow the corresponding installation steps above with the following change:

```bash
pip install .[tune]
```

### Test run

Upon successful installation, you will have the `pst` executable to train, tune, and predict. There are also other modules included as utilties that you can see using `pst -h`.

You will need to first download a trained vPST model:

```bash
pst download --trained-models
```

This will download both vPST models into `./pstdata`, but you can change the download location using `--outdir`.

You can use the test data for a test prediction run:

```bash
pst predict \
    --file test/test_data.graphfmt.h5 \ # this is in the git repo
    --checkpoint pstdata/pst-small_trained_model.ckpt \
    --outdir test_run
```

The results from the above command are available at `test/test_run/predictions.h5`. This test run takes fewer than 1 minute using a single CPU.

If you are unfamiliar with `.h5` files, you can use `pytables` (installed with PST as a dependency) to inspect `.h5` files in python, or you can install `hdf5` and use the `h5ls` to inspect the fields in the output file.

There should be 3 fields in the prediciton file:

1. `attn` which contains the per-protein attention values (shape: $N_{prot} \times N_{heads}$)
2. `ctx_ptn` which contains the contextualized PST protein embeddings (shape: $N_{prot} \times D$)
3. `genome` which contains the PST genome embeddings (shape: $N_{genome} \times D$)
    - Prior to version `1.2.0`, this was called `data`.

## Data availability

All data associated with the initial training model training can be found here: [https://doi.org/10.5061/dryad.d7wm37q8w](https://doi.org/10.5061/dryad.d7wm37q8w)

We have provided the README to the DRYAD data repository to render [here](DRYAD_README.md). Additionally, we have provided a programmatic way to access the data from the command line using `pst download`:

**NOTE**: we have recently changed the DRYAD repository corresponding to manuscript resubmission, so these commands will not work at the moment. However, the latest dataset will be available to download directly through DRYAD soon.

```txt
usage: pst download [-h] [--all] [--outdir PATH] [--esm-large] [--esm-small] [--vpst-large] [--vpst-small] [--genome] [--genslm]
                    [--trained-models] [--genome-clusters] [--protein-clusters] [--aai] [--fasta] [--host-prediction] [--no-readme]
                    [--supplementary-data] [--supplementary-tables]

help:
  -h, --help            show this help message and exit

DOWNLOAD:
  --all                 download all files from the DRYAD repository (default: False)
  --outdir PATH         output directory to save files (default: ./pstdata)

EMBEDDINGS:
  --esm-large           download ESM2 large [t33_150M] PROTEIN embeddings for training and test viruses (esm-large_protein_embeddings.tar.gz)
                        (default: False)
  --esm-small           download ESM2 small [t6_8M] PROTEIN embeddings for training and test viruses (esm-small_protein_embeddings.tar.gz)
                        (default: False)
  --vpst-large          download vPST large PROTEIN embeddings for training and test viruses (pst-large_protein_embeddings.tar.gz) (default:
                        False)
  --vpst-small          download vPST small PROTEIN embeddings for training and test viruses (pst-small_protein_embeddings.tar.gz) (default:
                        False)
  --genome              download all genome embeddings for training and test viruses (genome_embeddings.tar.gz) (default: False)
  --genslm              download GenSLM ORF embeddings (genslm_protein_embeddings.tar.gz) (default: False)

TRAINED_MODELS:
  --trained-models      download trained vPST models (trained_models.tar.gz) (default: False)

CLUSTERS:
  --genome-clusters     download genome cluster labels (genome_clusters.tar.gz) (default: False)
  --protein-clusters    download protein cluster labels (protein_clusters.tar.gz) (default: False)

MANUSCRIPT_DATA:
  --aai                 download intermediate files for AAI calculations in the manuscript (aai.tar.gz) (default: False)
  --fasta               download protein fasta files for training and test viruses (fasta.tar.gz) (default: False)
  --host-prediction     download all data associated with the host prediction proof of concept (host_prediction.tar.gz) (default: False)
  --no-readme           download the DRYAD README (README.md) (default: True)
  --supplementary-data  download supplementary data directly used to make the figures in the manuscript (supplementary_data.tar.gz) (default:
                        False)
  --supplementary-tables
                        download supplementary tables (supplementary_tables.zip) (default: False)
```

For flags relating to the download of specific files, you can add as many flags as you like.

### Model information

Specifically at DRYAD link, `trained_models.tar.gz` contains both sizes of the vPST foundation model, `pst-small` and `pst-large`. Each model was trained with the same input data.

The training and test data are also available in the above data repository.

Here is a summary of each model:

| Model       | # Encoder layers | # Attention heads | # Params | Embedding dim |
| :---------- | :--------------- | :---------------- | :------- | :------------ |
| `pst-small` | 5                | 4                 | 5.4M     | 400           |
| `pst-large` | 20               | 32                | 177.9M   | 1280          |

## Usage, Finetuning, and Model API

Please read the [wiki](https://github.com/AnantharamanLab/protein_set_transformer/wiki) for more information about how to use these models, extend them for finetuning and transfer learning, and the specific model API to integrate new models into your own workflows. **Note: This is still a work in progress.**

## Expected runtime and memory consumption

The expected runtime for training the final models after hyperparameter tuning can be found in `Supplementary Table 11` and ranged from 3.9-33.7h on 1 A100 GPU.

### Inference times

These are estimates of inference times for a dataset composed of ~12k viral genomes encoding ~140k proteins (such as the MGnify test dataset):

| Model Size | Accelerator | ESM2 embedding* | PST inference | Total Time |
|------------|-------------|-----------------|---------------|------------|
| Large      | 1 A100 GPU  | 18 min          | <1 min        | ~19 min    |
| Large      | 128 CPUs    | 6h              | <1 min        | ~6h        |
| Large      | 8 CPUs      | 96h             | 11 min        | ~96h       |
| Small      | 1 A100 GPU  | 9 min           | <1 min        | ~9 min     |
| Small      | 128 CPUs    | 3h              | <1 min        | ~3h        |
| Small      | 8 CPUs      | 48h             | 6 min         | ~48h       |

\* ESM2 embeddings are computed independently for each protein, so input FASTA files can be split into equal batches and processed in parallel with as many GPUs as available.

- These will need to be concatenated in the same order as the original FASTA file.

### Memory

Memory usage should be negligible for inference, especially if using a `LazyGenomeDataset`. Less than 4GB of memory is needed for inference.

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

### Citation

Please cite our preprint if you find our work useful:

Martin C, Gitter A, Anantharaman K. (2024) "[Protein Set Transformer: A protein-based genome language model to power high diversity viromics.](https://doi.org/10.1101/2024.07.26.605391)"

```bibtex
@article {
  author = {Cody Martin and Anthony Gitter and Karthik Anantharaman},
  title = {Protein Set Transformer: A protein-based genome language model to power high diversity viromics},
  elocation-id = {2024.07.26.605391},
  year = {2024},
  doi = {10.1101/2024.07.26.605391},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/10.1101/2024.07.26.605391v1},
  eprint = {https://www.biorxiv.org/content/10.1101/2024.07.26.605391v1.full.pdf}
  journal = {bioRxiv},
}
```
