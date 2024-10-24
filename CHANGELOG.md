# Change log since first stable release (v1.0)

## v1.3.1 - 2024-10-24

### Fixed

- `GenomeDataset`
  - There was a bug with fragmenting single-scaffold genomes. If the fragment size created a sub-scaffold fragment that only contained 1 protein, this will lead to an error when creating the edge index since this genome was not originally multi-scaffold.
  - The dataset now keeps track of if fragmentation occurred to allow either multi-scaffold genomes or fragmented genomes to have a scaffold or fragment with 1 protein.

## v1.3.0 - 2024-10-23

### Added

- `pst graphify` now adds information about a multi-scaffold genomes to the graph-formatted h5 file.
  - Datasets where there are multi-scaffold genomes will have the `genome_label` field in the h5 file.
  - This also can take in a tsv file of FASTA/ESM file pairs to individually convert each pair to a dataset before merging them all into a single h5 file. This is probably most useful for a directory of MAGs embed individually.
- `GenomeDataset`
  - To keep track of multi-scaffold genomes more explicitly, there are 2 new attributes called `genome_is_multiscaffold` and `scaffold_part_of_multiscaffold`.
    - These adhere to the new feature-level explicit attributes names, so `genome_is_multiscaffold` is a boolean tensor of shape `[num_genomes]` where True means the genome contains multiple scaffolds. `scaffold_part_of_multiscaffold` is an expanded version that labels each scaffold as belonging to a multi-scaffold genome or not.
  - Added properties that compute the number of proteins, scaffolds, and genomes. They are all named `.num_X` where `X` is either "proteins", "scaffolds", or "genomes".
    - Further, `.num_proteins_per_genome` returns a tensor of shape `[num_genomes]` that totals all proteins from all scaffolds for each genome

### Changed

- `GenomeDataset`
  - All major attributes of this class were renamed to be prefixed with either protein, scaffold, or genome to indicate what biological level the internal tensors referred to.
    - For example, previously the protein embedings were stored in the `data` field and are now stored in the `protein_data` field.
    - Further, these prefixes indicate the expected shape of each tensor.
    - Properties were added to point old names to these new fields for backwards compatibility. These are all deprecated, which will be indicated in an IDE and at runtime if these are used.
    - Likewise, registered features must be labeled as either "protein", "scaffold", or "genome" and be the correct shape.
  - Modified reading data from the source graph-formatted h5 file
    - The only required fields are `data`, `sizes`, and `strand`, which refer to the stacked protein embeddings, number of proteins per scaffold, and the strand of each protein, respectively.
    - Other "top-level" attributes in the h5 file are `ptr`, `scaffold_label`, and `genome_label`, which refer to the CSR index pointer for random access of all protein embeddings from a source scaffold, the per-scaffold IDs (not generally useful for users to set this), and the per-scaffold genome IDs (for multi-scaffold genomes), respectively.
      - These will be loaded into direct attributes on the dataset object if present. Otherwise, they will be computed (`ptr`) or set to default values (`genome_label` and `scaffold_label`). For example, the `ptr` field will be stored in the `.scaffold_ptr` attribute.
    - All other arrays in the h5 file will be treated as registered features. There are now 2 options to set this up in the h5 file:
      - Prepend the name of the attribute with "protein", "scaffold", or "genome".
      - Create groups (basically like subdirectories) in the file called "protein", "scaffold", "genome". All arrays under each group will be appropriately registered.
      - NOTE: Registered features are registered without overwriting, so all names must be unique in the same feature level. Ie, there cannot be 2 `featureA`s under the "scaffold" level, but there can be a scaffold-level `featureA` and a genome-level `featureA`.
  - `class_id` and `weights` are no longer default fields since these are training/objective specific and should not be automatically part of a general-purpose dataset class.
    - If present in the h5 file, these become registered features. There is special handling of `class_id` to interpret this as a scaffold-level feature.
    - There are deprecated helper properties to try to grab these features if they were registered. If they were not, `None` will be returned.
  - Changed `.any_genomes_have_multiple_scaffolds` method to `any_multi_scaffold_genomes` for simplicity without sacrificing descriptiveness.
- `GenomeDataModule`
  - Change references to `GenomeDataset` attributes to be consistent with new name changes
  - Currently requires `class_id` in the h5 file if training/finetuning since this was required by the PST trained in the manuscript. Should either decouple this or indicate to users that this `GenomeDataModule` should be modified/subclassed for other objectives. TODO: Could create a base datamodule.
    - For inference, `class_id` is not required.
    - Registers `weights` on the `GenomeDataset` if `class_id` is present

### Fixed

- `GenomeDataset`
  - There was an error creating the edge index for multi-scaffold genomes that encode more than 1 protein where a single scaffold only contains 1 protein. This is a rare case (less than 0.5% of test vMAGs from IMG/VR v4), but we allow scaffolds containing 1 protein ONLY if they are part of multi-scaffold genomes.
  - Note: For the underlying `GenomeGraph`, this will raise a `ValueError`, so it is up to the caller to handle this.

## v1.2 - 2024-10-16

### Added

- `GenomeDataset`
  - Add support for multi-scaffold genomes
    - The default view of genomes is that they are all a single contiguous segment (scaffold).
    - Thus, the dataloaders are all inherently scaffold loaders
    - Added `scaffold_label` and `genome_label` as major fields in the dataset object.
      - These can also be read from the h5 file, but if they are not present, the genomes will be considered as single scaffold.
      - The more important addition to the h5 file would be `genome_label` for multi-scaffold genomes, but `scaffold_label` could be added if data are pre-fragmented for some reason.
    - Can check if there are any multi-scaffold genomes with `GenomeDataset.any_genomes_have_multiple_scaffolds()`
    - Added a `GenomeDataLoader` to load all scaffolds or fragments of multi-scaffold genomes instead of individual scaffolds or fragments.
      - The default data loader is an individual scaffold loader, which may not be useful for certain tasks or datasets that involve multi-scaffold genomes.
  - Enable artificial genome fragmentation (`GenomeDataset.fragment`)
    - This is useful for reducing memory burdens further for very large genomes that encode thousands of proteins like bacteria, but also solves the challenge of embedding genomes that encode more proteins than the model was trained for.
    - The implementation involves a simple regrouping of the proteins into different sub-boundaries for each scaffold.
      - Each fragment is then viewed as a separate scaffold and **embed individually** by PSTs
        - The new `scaffold_label` and `genome_label` tensors can be used to reduce the fragment-level embeddings into scaffold or genome level embeddings **AFTER** all fragments are embed.
      - Protein-protein edges are only allowed between what are considered "contiguous" genomic segments, meaning that these are only created between these artificial fragments.
    - For callers, the dataset can either be pre-fragmented upon loading the dataset (`--fragment-size` at command line) or in response to being too large for a specific model.
      - Added `fragment_size` to the `DataConfig` for pre-fragmentation
- Finetuning mode
  - This is exclusively for updating the model weights for pretrained PSTs with new genomes. (Currently only for the genomic PST)
  
### Changed

- Prediction/inference mode
  - Previously, the "scaffold" level embeddings (also called graph embeddings), were saved in the output h5 file under the `data` node.
  - These are now changed to `fragment`, `scaffold`, and `genome` to distinguish between artificial genomic fragments, individual scaffolds, and entire genomes (that may include multi-scaffold genomes).
    - Depending on the original data and if fragmenting will be used, one or multiple of these fields will be available. `genome` will always be present, and `scaffold` will only be present if multi-scaffold genomes are in the dataset. Similary, `fragment` is only present if genome fragmenting was enabled.
    - Lower level embeddings are always averaged to produce the embedding for the next level. For example, the embedding for single scaffold split into 3 fragments will be the average of those 3 scaffold fragments. Likewise, the genome embedding for multi-scaffold genome with 3 scaffolds will be the average of those 3 scaffolds.
      - For genomes without fragments or scaffolds, the higher level embedding may be the same as the lower level embedding since there is nothing to average.
- Activated logger when using command line
- Default accelerator is now auto resolved instead of being a GPU.
- `GenomeDataset` can be sliced and indexed with multiple indices instead of just a single integer

## v1.1 - 2024-10-09

### Added

- `GenomeDataModule` and `GenomeDataset`
  - Can register new features that are not natively part of the genome graph formatting.
  - This enables changing objectives to, for example, a classification problem with known labels.
- Command line:
  - `pst -v` or `pst --version` will now print the version of the `pst` module installed.

### Fixed

- Base PST classes:
  - Loading from pretrained models has been correctly implemented. This allows for loading a pretrained genomic PST (ie a subclass of `BaseProteinSetTransformer` to use for a protein-only task, such as for a subclass of `BaseProteinSetTransformerEncoder`).
    - This also correctly handles the definition of new learnable layers when loading a pretrained checkpoint.
    - The `from_pretrained` class method works for regular checkpoints and loading a different pretrained model.
  - The public classes for subclassing PSTs are now `BaseProteinSetTransformer` and `BaseProteinSetTransformerEncoder` to be subclassed for genome- (or dual genome-protein) or protein-level objectives, respectively.

## v1.0 - 2024-10-08

### Added

- Introduced base classes to allow customizing the objective function:
  - `BaseProteinSetTransformer` is the primary entrypoint for the underling `SetTransformer` model
  - `ProteinSetTransformer` defines a forward pass with triplet sampling and point swap sampling for a triplet loss objective
    - This is the model that was pretrained in the manuscript
  - `BaseProteinSetTransformerEncoder` is used to create models focused only on the protein aspect without genome decoding
  - `ProteinSetTransformerEncoder` is a specific implementation of a `BaseProteinSetTransformerEncoder` taht also uses a triplet loss objective but without augmentation
  - These can all be loaded from pretrained checkpoints with the `from_pretrained` class method.
    - The `GenomeDataModule` can also be loaded from a pretrained checkpoint with a class method of the same name, since there are data-formatting-specific tunable hyperparameters.
  - These models all use a `pydantic.BaseModel` config called `BaseModelConfig`. This can be customized with a custom subclass of the `BaseLossConfig` to adjust model- and objective-specific stateful hyperparameters.
- `ModelConfig`:
  - Added `max_proteins` to enable users to change the maximum allowed genome size for the positional embedding LUT. This can be controled at the command line.
- Public API was finalized to enable simple imports for end users

### Fixed

- Set model to `eval` mode during inference
