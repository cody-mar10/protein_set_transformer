# PST manuscript methods

The PST manuscript methods section map to different repositories or files located in this main repository:

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
- [Graph-based host prediction framework](https://github.com/cody-mar10/PST_host_prediction)
- [Constructing the virus-host interaction network](https://github.com/cody-mar10/PST_host_prediction/blob/main/data/create_knowledge_graph.ipynb)
  - Specific knowledge graphs can be found [here](https://github.com/cody-mar10/PST_host_prediction/blob/main/data/knowledge_graphs)
- [Host prediction model evaluation](https://github.com/cody-mar10/PST_host_prediction/blob/main/evaluations/iPHoP_test_set_evaluation.ipynb)
  - [Choosing the best models](https://github.com/cody-mar10/PST_host_prediction/blob/main/evaluations/choosing_best_models.ipynb)