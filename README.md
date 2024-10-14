# DiGNNExplainer

In the paper, <strong>Synthesizing Node Features for Model-Level Explanations of
Heterogeneous Graph Neural Networks</strong>, we explain GNNs on a model-level by synthetically generating explanation graphs with node features from the underlying heterogeneous data.

## Environment setup

The specifications of the machine used to run the code:
 - OS: `Ubuntu 22.04.2 LTS`
 - CPU: `AMD Ryzen 9 5900HX`
 - GPU: `NVIDIA GeForce RTX 3070`
   
1. For jupyter notebooks, create a conda environment using the environment.yml<br/>

   ```
   conda env create --name DiGNNExplainer --file environment.yml
   ```  
   Or
 
 - Create a conda environment using the following commands:  
 
   `conda create -n DiGNNExplainer python=3.10.6 `

   `conda activate DiGNNExplainer`  

   `conda install pytorch==2.1.2  pytorch-cuda=12.1 -c pytorch -c nvidia`


- Run `pip install package-name` for the following packages:
   
    `notebook` 

    `torch-geometric==2.4.0`  

    `matplotlib==3.5.3` 

    `pandas==1.5.3` 

    `python-louvain==0.16` 

    `seaborn==0.12.2` 

    `dgl==1.1.3` 

    `import-ipynb`

    `littleballoffur==2.3.1`

    	
 2.  For `diffusion_graph_gen` and `diffusion_node_feature_gen` create a conda enviroment by following the installation steps of DiGress: https://github.com/cvignac/DiGress.  <br/>
 
 3.  For `baseline_vae`, refer to https://github.com/deepfindr/gvae.

## Run code
1. To run all jupyter notebooks, 
- Activate conda environment:
 `conda activate DiGNNExplainer`

- Run jupyter notebook:
 `jupyter notebook`

- Add references to code for the sampled features for IMDB dataset using TabDDPM from [here](https://drive.google.com/file/d/1cYWwO4WgfafH3G0bOw69DQsLAFUiJs-5/view?usp=sharing).<br/>       
- Use the `.dat` files for PubMed dataset wherever `.dat` files are present in code, e.g. `PubMed/node.dat`, `PubMed/link.dat`, `PubMed/label.dat.test`

2. For graph generation and node feature generation you need to run the code for each node size and each node feature size.<br/>
- To run `baseline_vae`:<br/>
	- Specify node size, node feature size, dataset in `baseline_vae/config.py` as: 
	  ```
	  MAX_NODES = 15
	  NODE_FEATURE_SIZE = 200
	  DATASET = "pubmed"   
	  ```
   	 	`NODE_FEATURE_SIZE` is the size of all features of the dataset.<br/>
      
	- Navigate to `baseline_vae` folder and run
       	 `python3 train.py`
   
- To run `diffusion_graph_gen`:<br/>
	- Specify dataset and node size in `diffusion_graph_gen/configs/general/general_default.yaml`:
 
	  ```
	  node_size: 15
	  dataset_name: 'dblp'
	  ```
	- Navigate to `diffusion_graph_gen` folder and run
	   `python3 main.py`

 - To run `diffusion_node_feature_gen` for DBLP Author class:<br/>
 	- Specify dataset in `configs/config.yaml`, and node class, node feature size in `diffusion_node_feature_gen/configs/dataset/dblp.yaml`:
    
	  ```
	  dataset: dblp
	  node_class: 0
	  node_feature_size: 4
	  ```
	- Navigate to `diffusion_node_feature_gen` folder and run
          `python3 main.py`
 <br/>
 
3. To run code in `baseline_explainers`,
- For `xgnn`, refer to https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN.
- For `gnninterpreter`, follow the installation steps of https://github.com/yolandalalala/GNNInterpreter/tree/main and refer to https://github.com/MeneerTS/FACT2024_GNNInterpreter/tree/main/GNNInterpreter-Most-Recent-Version. 
- For `d4explainer`, follow the installation steps of https://github.com/Graph-and-Geometric-Learning/D4Explainer/tree/main.

 <br/>
4. For node feature generation using TabDDPM refer to https://github.com/yandex-research/tab-ddpm.



   
       
## Source of datasets
- DBLP : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html
- PubMed : https://github.com/yangji9181/HNE/tree/master/Data
- IMDB : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.IMDB.html
- MUTAG : https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.TUDataset.html
- BA-Shapes : https://docs.dgl.ai/en/1.1.x/generated/dgl.data.BAShapeDataset.html
- Tree-Cycle : https://docs.dgl.ai/en/1.1.x/generated/dgl.data.TreeCycleDataset.html
- Tree-Grids : https://docs.dgl.ai/en/0.9.x/generated/dgl.data.TreeGridDataset.html
- BA-3Motif : https://github.com/Wuyxin/ReFine/tree/main/data/BA3/raw

