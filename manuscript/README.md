# PST manuscript methods

The PST manuscript methods section map to different repositories or files located in this main repository. Files referenced in these notebooks are either located in the DRYAD repository (datasets, supplementary data, or supplementary tables). The supplementary tables may also be found associated with the manuscript itself.

The files should have the same names. However, due to the combined sized of all `datasets/` files (>170GB), these files are individually grouped into subgroups in the DRYAD repository. The specific file names are the same as referenced in these notebooks, but the DRYAD `README` will tell you what specific tarball you need.

Be warned that the memory requirements of some of these analyses can reach up to 1TB if you try to reproduce these analyses with the full datasets.

## Table of contents

- [ESM2 protein language model embeddings](https://github.com/cody-mar10/esm_embed/tree/main)
- [Modified Leave-One-Group-Out cross validation and hyperparameter tuning](https://github.com/cody-mar10/lightning-crossval)
  - Part of the specific implementation is also found [here](../src/pst/training/)
- [GenSLM open reading frame (ORF) and genome embeddings](https://github.com/cody-mar10/genslm_embed)
- [Hyena-DNA genome embeddings](https://github.com/cody-mar10/hyena-dna-embed)
- [Tetranucleotide frequency vectors as simple genome embeddings](genome_embeddings/kmer.ipynb)
- Clustering genome and protein embeddings
  - [Genomes](genome_embeddings/clustering.ipynb)
  - [Proteins](protein_embeddings/clustering.ipynb)
- Genome and protein clustering evaluation
  - [Genome viral and host taxonomy purity](genome_embeddings/evaluations/taxonomy_purity.ipynb)
  - [Protein functional purity](protein_embeddings/evaluations/functional_purity.ipynb)
- [Average amino acid identity (AAI)](genome_embeddings/evaluations/AAI.ipynb)
  - Averaging AAI over each genome cluster found [here](genome_embeddings/evaluations/average_AAI_per_cluster.ipynb)
- [Average amino acid identity (AAI) genome clustering](genome_embeddings/evaluations/AAI.ipynb)
- Protein functional annotation
  - [Re-categorizing VOG](protein_embeddings/annotations/relabel_VOG.ipynb)
- [Protein attention scaling and analysis](protein_embeddings/evaluations/attention.ipynb)
- [Protein annotation improvement](protein_embeddings/evaluations/annotation_improvement.ipynb)
- [Protein function co-clustering](protein_embeddings/evaluations/function_co-clustering.ipynb)
- [Protein functional module detection](protein_embeddings/evaluations/functional_module_detection.ipynb)
- [Capsid structure searches](protein_embeddings/evaluations/capsid_structure_searches.ipynb)
- [Graph-based host prediction framework](https://github.com/cody-mar10/PST_host_prediction)
- [Constructing the virus-host interaction network](https://github.com/cody-mar10/PST_host_prediction/blob/main/data/create_knowledge_graph.ipynb)
  - Specific knowledge graphs can be found [here](https://github.com/cody-mar10/PST_host_prediction/blob/main/data/knowledge_graphs)
- [Host prediction model evaluation](https://github.com/cody-mar10/PST_host_prediction/blob/main/evaluations/iPHoP_test_set_evaluation.ipynb)
  - [Choosing the best models](https://github.com/cody-mar10/PST_host_prediction/blob/main/evaluations/choosing_best_models.ipynb)
