# DiGNNExplainer

In the paper, <strong>Synthesizing Node Features for Model-Level Explanations of
Heterogeneous Graph Neural Networks</strong>, we explain GNNs on a model-level by synthetically generating explanation graphs with node features from the underlying heterogeneous data.

## Instructions to run code

1. To run all jupyter notebooks, create a conda environment
   
   conda create --name env --file requirements.txt
   <br>
   conda activate env
   <br>
   jupyter notebook

2. To run code for diffusion_graph_gen and diffusion_node_feature_gen, follow the installation steps of DiGress: https://github.com/cvignac/DiGress
3. To run code for baseline_vae, follow the installation steps of https://github.com/deepfindr/gvae
4. Add references to code for the original and sampled graphs, and node features from the folder location:
   graph generator/diffusion models/ and the sampled features for IMDB using TabDDPM from [here](https://drive.google.com/file/d/1cYWwO4WgfafH3G0bOw69DQsLAFUiJs-5/view?usp=sharing).

## Source of datasets
- DBLP : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html
- PubMed : https://github.com/yangji9181/HNE/tree/master/Data
- IMDB : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.IMDB.html





