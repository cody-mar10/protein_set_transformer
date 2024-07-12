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

Specifically at the above link, `trained_models.tar.gz` contains both sizes of the vPST foundation model, `pst-small` and `pst-large`.

The training and test data are also available in the above data repository.
