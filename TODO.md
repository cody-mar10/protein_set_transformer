# TODO

## Model

- Technically the weighted average of proteins has an additional division operation
  - Not sure if this matters that much since the embedding values are constantly being normalized
  - Would also require retraining the model
- Could train a final model with both training and test datasets for true downstream use
- Should we consider averaging over scaffolds instead of the entire genome?
  - Technically the protein IDs are specific to the scaffold (for multi-scaffold viruses), and not contiguously numbered for the entire genome
  - The easiest fix is keep track of both the scaffold as well as the entire genome
    - This probably requires to different index pointers for each
  - Then proteins can attend only within each scaffold
  - Then proteins can be aggregated over each scaffold and then again over all scaffolds in each genome
- Huggingface integration

## Package

- This would be so much easier to use as a pypi package, but pypi does not like that I have my custom fork of `pydantic-argparse` as a requirement
  - Need to resolve this, perhaps using a submodule

## User experience

- Should add the following utilities:
  - Pipeline to go directly from FASTA files to PST outputs
    - This will be challenging since each requires different python/pytorch versions
  - Script to reformat protein embeddings into the graph format
