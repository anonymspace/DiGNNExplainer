# DiGNNExplainer

In the paper, <strong>Discrete Diffusion-Based Model-Level Explanation of Heterogeneous GNNs with Node Features</strong>, we explain GNNs on a model-level by synthetically generating explanation graphs with node features from the underlying heterogeneous data.

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
     
     `pykeen`
     
     `easydict`
     
     `rdkit`

    	
 2.  For `diffusion_graph_gen` and `DiTabDDPM` create a conda enviroment by following the installation steps of DiGress: https://github.com/cvignac/DiGress.  <br/>
 
 3.  For `baseline_vae`, refer to https://github.com/deepfindr/gvae.

## Run code
1. To run all jupyter notebooks, 
- Activate conda environment:
 `conda activate DiGNNExplainer`

- Run jupyter notebook:
 `jupyter notebook`


2. For graph generation and node feature generation,<br/>
- To run `baseline_vae`:<br/>
	- Specify node feature size, dataset in `baseline_vae/config.py` as: 
	  ```
	  NODE_FEATURE_SIZE = 3066
	  DATASET = "imdb"   
	  ```
   	 	`NODE_FEATURE_SIZE` is the size of all features of the dataset.<br/>
      
	- Navigate to `baseline_vae` folder and run
       	 `python3 train.py`
   
- To run `diffusion_graph_gen`:<br/>
	- Specify dataset in `diffusion_graph_gen/configs/general/general_default.yaml`:
 
	  ```
	  dataset_name: 'dblp'
	  ```
	- Navigate to `diffusion_graph_gen` folder and run
	   `python3 main.py`

 - To run `DiTabDDPM` for DBLP Author class:<br/>
 	- Specify dataset in `configs/config.yaml`, and node class, node feature size in `diffusion_node_feature_gen/configs/dataset/dblp.yaml`:
    
	  ```
	  dataset: dblp
	  node_class: 0
	  node_feature_size: 4
	  ```
	- Navigate to `diffusion_node_feature_gen` folder and run
          `python3 main.py`
          
 - To run original `TabDDPM` for DBLP Author class 0:<br/>
 	- Refer to [TabDDPM](https://github.com/yandex-research/tab-ddpm) for data preparation, and sample node features by running `pipeline.py` with the parameters:
    
	  ```
	  --config exp/author0/ddpm_cb_best/config.toml --train --sample
	  ```
          
 <br/>      		


       
## Source of datasets
- DBLP : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html
- IMDB : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.IMDB.html
- MUTAG : https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.TUDataset.html
- BA-Shapes : https://docs.dgl.ai/en/1.1.x/generated/dgl.data.BAShapeDataset.html
- Tree-Cycle : https://docs.dgl.ai/en/1.1.x/generated/dgl.data.TreeCycleDataset.html
- Tree-Grids : https://docs.dgl.ai/en/0.9.x/generated/dgl.data.TreeGridDataset.html
- BA-3Motif : https://github.com/Wuyxin/ReFine/tree/main/data/BA3/raw


