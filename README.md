# Protein Set Transformer

This repository contains the Protein Set Transformer (PST) framework for contextualizing protein language model embeddings at genome-scale to produce genome embeddings. You can use this code to train your own models. Using our foundation model pre-trained on viruses (vPST), you can also generate genome embeddings for input viruses.

For more information, see our manuscript:

Protein Set Transformer: A protein-based genome language model to power high diversity viromics.  
Cody Martin, Anthony Gitter, and Karthik Anantharaman.
*bioRxiv*, 2024, doi: [10.1101/2024.07.26.605391](https://doi.org/10.1101/2024.07.26.605391).

## Installation

>We highly recommend using [uv](https://docs.astral.sh/uv/) for installation, since it will be significantly faster to solve the dependencies and install everything.
>
>If you don't have the ability to install `uv`, then just remove `uv` from the following commands.

### Optional: Setup a virtual environment

If you wish, you can setup a virtual environment to install the PST dependencies into using, for example, `conda`, `mamba`, or `pyenv`:

```bash
mamba create -n pst -c conda-forge 'python>=3.9'
```

Just make sure you activate your virtual environment before proceeding with the installation.

### Basic installation

You can try simply doing:

```bash
uv pip install torch
uv pip install ptn-set-transformer --no-build-isolation
```

This will do 2 things:

1. Install the latest version of `PyTorch` with the default `CUDA` runtime, even if your system does not have GPUs
   1. This will run fine on CPU-only systems, but the install will be larger
2. Install the `PST` library and force some of the required `PyTorch` extension libraries (specifically `PyTorch-Scatter`) to build on your target machine. This will take a few minutes.

For most use cases, this should work fine.

-----

Optional Note: If you would like to install the latest release from this repository, you will likely need to link your git command line interface with an online github account. Follow [this link](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git#setting-up-git) for help setting up git at the command line.

*If you would like to proceed further for a more advanced setup or ran into issues, then try the more manual setups below.*

### Manually setup PyTorch

You can check the installation pages for [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Since `PyTorch > 2.5.0`, `conda` is no longer an option to install `PyTorch`, and we have uncapped the `PyTorch` version since the minor updates will not affect PST. Thus, these examples will show `pip`/`uv pip`.

#### Without GPUs

##### 1. Install PyTorch

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

##### 2. Install PyTorch extension libraries

Then, depending on the version of `PyTorch` installed, the following command needs to be updated to install the `PyTorch` extension libraries (`PyTorch-Geometric`, `PyTorch-Scatter`, `PyTorch-Sparse`):

```bash
uv pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-{TORCHVERSION}+cpu.html
```

where `{TORCHVERSION}` is replaced with the specific `PyTorch` version (ie: `2.8.0`)

##### 3. Install PST library

```bash
uv pip install ptn-set-transformer
```

#### With GPUs

##### 1. Install PyTorch

```bash
uv pip install torch
```

This will install the `PyTorch` library with the default `CUDA` runtime.

If you wish to download a precompiled `PyTorch` library with a different `CUDA` runtime, then you can adjust the command to be:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/{CUDAVERSION}
```

where `{CUDAVERSION}` is a supported `CUDA` version of the latest `PyTorch` release (i.e. `cu129`).

Note the `{CUDAVERSION}` of the `PyTorch` library installed since it will be needed for the next step.

##### 2. Install PyTorch extension libraries

Then, depending on the version of `PyTorch` installed, the following command needs to be updated to install the `PyTorch` extension libraries (`PyTorch-Geometric`, `PyTorch-Scatter`, `PyTorch-Sparse`):

```bash
uv pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-{TORCHVERSION}+{CUDAVERSION}.html
```

where `{TORCHVERSION}` is replaced with the specific `PyTorch` version (ie: `2.8.0`) and `{CUDAVERSION}` is the `CUDA` version of the installed `PyTorch` release (i.e. `cu129`)

##### 3. Install PST library

```bash
uv pip install ptn-set-transformer
```

### Installation issues

Due to the various `PyTorch` dependencies, which are typically shipped as precompiled binaries for specific Python/CUDA/GCC/Linux/etc versions, there can sometimes be version conflict issues that can be hard to resolve.

We have primarily encountered these errors when installing the `PyTorch` extension libaries, so we will focus on how to resolve issues installing `torch_geometric`, `torch_scatter`, and `torch_sparse`.

#### CPU/GPU compatibility errors

If `PyTorch` was installed CPU-only, then the extension libraries also need to be installed CPU-only.

Ensure that they are installed like this:

```bash
uv pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-{TORCHVERSION}+cpu.html
```

where `{TORCHVERSION}` is replaced with the specific `PyTorch` version (ie: `2.8.0`)

Conversely, if `PyTorch` is installed with GPU support, then the extension libraries also need to be installed with GPU support corresponding to the same `PyTorch` version and `CUDA` runtime version. If you are unsure about this, then you can obtain this information like this:

```bash
python -c 'import torch; print(torch.__version__)'
```

which will return a string such as `2.8.0+cu126` or `2.8.0+cpu`.

#### GLIBC version errors

The precompiled binaries are compiled with specific versions of your system's C compiler, which may not be present on your system. You could update your C compiler/C lib or install a version that is compatible with the precompiled binaries. However, it is much simpler to recompile these libraries for your target system:

```bash
uv pip install torch_geometric torch_scatter torch_sparse --verbose --no-build-isolation
```

This will take several minutes.

### Installing for training a new PST

We implemented a hyperparameter tuning cross validation workflow implemented using [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) in a base library called [lightning-crossval](https://github.com/cody-mar10/lightning-crossval). Part of our specific implementation for hyperparameter tuning is also implemented in the PST library.

The latest versions of `PST` requires the tuning dependencies, so there is no additional required installed.

### Test run

Upon successful installation, you will have the `pst` executable to train, tune, and predict. There are also other modules included as utilties that you can see using `pst -h`.

You will need to first download a trained vPST model:

```bash
pst download --model.choices="[PST-TL-P__small]"
```

This will download both vPST models into `./pstdata`, but you can change the download location using `--outdir`.

You can use the test data for a test prediction run:

```bash
pst predict \
    --file examples/sample_data.graphfmt.h5 \
    --predict.checkpoint pstdata/PST-TL-P__small.ckpt \
    --predict.output PST-TL-P__small_predictions.h5
```

The results from the above command are available at `examples/PST-TL-P__small_predictions.h5`. This test run takes fewer than 1 minute using a single CPU.

If you are unfamiliar with `.h5` files, you can use `pytables` (installed with PST as a dependency) to inspect `.h5` files in python, or you can install `hdf5` and use the `h5ls` to inspect the fields in the output file.

There should be 3 fields in the prediciton file:

1. `attn` which contains the per-protein attention values (shape: $N_{prot} \times N_{heads}$)
2. `ctx_ptn` which contains the contextualized PST protein embeddings (shape: $N_{prot} \times D$)
3. `genome` which contains the PST genome embeddings (shape: $N_{genome} \times D$)
    - Prior to version `1.2.0`, this was called `data`.

## What if I don't have GPU access?

We have provided a [PST inference notebook](https://colab.research.google.com/github/cody-mar10/protein_set_transformer/blob/main/examples/pst_inference.ipynb) that can be used within a `Google Colab` runtime environment. You can use free (although less powerful and lower memory) GPUs for inference of relatively small datasets (ie <10k genomes encoding <250k proteins).

## Data availability

All data associated with the initial training model training can be found here: [https://doi.org/10.5061/dryad.d7wm37q8w](https://doi.org/10.5061/dryad.d7wm37q8w)

We have provided the [README to the DRYAD data repository to render here](DRYAD_README.md). Additionally, we have provided a programmatic way to access the data from the command line using `pst download`:

**NOTE**: we have recently changed the DRYAD repository corresponding to manuscript resubmission, so these commands will not work at the moment. However, the latest dataset will be available to download directly through DRYAD soon.

```txt
usage: pst [options] download [-h] [--config CONFIG] [--print_config[=flags]] [--manuscript CONFIG]
                              [--manuscript.choices CHOICES] [--cluster CONFIG] [--cluster.choices CHOICES]
                              [--model CONFIG] [--model.choices CHOICES] [--embeddings CONFIG]
                              [--embeddings.choices CHOICES] [--all {true,false}] [--outdir OUTDIR]

Download mode to download data and trained models from DRYAD. Example usage: pst download
--manuscript.choices="[source_data, supplementary_data]"

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags
                        customizes the output and are one or more keywords separated by comma. The supported
                        flags are: comments, skip_default, skip_null.
  --all {true,false}    Download all data from the DRYAD repository. (type: bool, default: False)
  --outdir OUTDIR       Output directory to save files. (type: <class 'Path'>, default: pstdata)

MANUSCRIPT DATA:
  --manuscript CONFIG   Path to a configuration file.
  --manuscript.choices CHOICES, --manuscript.choices+ CHOICES
                        Download manuscript-specific data. Defaults to only the README. (type:
                        list[Literal['source_data', 'supplementary_data', 'supplementary_tables',
                        'host_prediction', 'fasta', 'foldseek_databases', 'README']] | None, default: null)

CLUSTER DATA:
  --cluster CONFIG      Path to a configuration file.
  --cluster.choices CHOICES, --cluster.choices+ CHOICES
                        Download genome or protein clusters. (type: list[Literal['genome', 'protein']] | None,
                        default: null)

TRAINED MODELS:
  --model CONFIG        Path to a configuration file.
  --model.choices CHOICES, --model.choices+ CHOICES
                        Download pretrained models. (type: list[Literal['PST-TL-P__small', 'PST-TL-P__large',
                        'PST-TL-T__small', 'PST-TL-T__large', 'PST-MLM']] | None, default: null)

EMBEDDINGS:
  --embeddings CONFIG   Path to a configuration file.
  --embeddings.choices CHOICES, --embeddings.choices+ CHOICES
                        Download embedding files. (type: list[Literal['esm2', 'IMGVR_PST-TL-P__large',
                        'IMGVR_PST-TL-P__small', 'IMGVR_PST-TL-T__large', 'IMGVR_PST-TL-T__small', 'MGnify_PST-
                        TL-P__large', 'MGnify_PST-TL-P__small', 'MGnify_PST-TL-T__large', 'MGnify_PST-TL-
                        T__small', 'genslm_ORF', 'train_PST-TL-P__large', 'train_PST-TL-P__small', 'train_PST-
                        TL-T__large', 'train_PST-TL-T__small', 'PST-TL_genome', 'other_genome']] | None,
                        default: null)
```

Example Usage:

You need to write your arguments like a Python list, all in quotes, which enables downloading multiple files at a time.

You still need to write the command this way even if you download 1 file.

```bash
pst download \
  --model.choices="[PST-TL-P__small, PST-TL-P__large]" \
  --manuscript.choices="[supplementary_tables]"
```

### Model information

The DRYAD repository contains all PST models pretrained on our viral genome dataset. Each model was trained with the same input data.

The training and test data are also available in the above data repository.

Here is a summary of each model:

| Model              | # Encoder layers | # Attention heads | # Params | Embedding dim |
| :----------------- | :--------------- | :---------------- | :------- | :------------ |
| `PST-TL-T__small`  | 5                | 4                 | 5.4M     | 400           |
| `PST-TL-T__large`  | 20               | 32                | 177.9M   | 1280          |
| `PST-TL-P__small`  | 5                | 4                 | 5.4M     | 400           |
| `PST-TL-P__large`  | 5                | 4                 | 21.3M    | 800           |
| `PST-MLM-T__small` | 5                | 4                 | 23.8M    | 960           |
| `PST-MLM-T__large` | 5                | 4                 | 93.6M    | 1920          |
| `PST-MLM-P__small` | 30               | 32                | 93M      | 960           |
| `PST-MLM-P__large` | 10               | 8                 | 185.8M   | 1920          |

The model name follows this format: `PST-OBJECTIVE-CV__ESMsize`, where:

- `OBJECTIVE` refers to the training objective
  - `TL` = triplet loss
  - `MLM` = masked language modeling
- `CV` refers to how the cross validation groups were defined
  - `P` = non overlapping protein diversity
  - `T` = viral taxonomic realm
- `ESMsize` refers to the relative size of ESM2 embeddings used to train each model, *not the size fo the PST model itself*
  - `large` = `esm2_t30_150M` (640 dim)
  - `small` = `esm2_t6_8M` (320 dim)

## Usage, Finetuning, and Model API

Please read the [wiki](https://github.com/AnantharamanLab/protein_set_transformer/wiki) for more information about how to use these models, extend them for finetuning and transfer learning, and the specific model API to integrate new models into your own workflows. **Note: This is still a work in progress. There is an [example Jupyter notebook provided](examples/finetuning.ipynb)**

## Expected runtime and memory consumption

The expected runtime for training the final models after hyperparameter tuning can be found in `Supplementary Table 4` and ranged from 3.9-33.7h on 1 A100 GPU.

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
