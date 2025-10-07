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
     
     `rdkit`
     
     `pyemd==1.0.0`
     
     `easydict==1.13`

    	
 2.  For `diffusion_graph_gen` and `DiTabDDPM` create a conda enviroment by following the installation steps of DiGress: https://github.com/cvignac/DiGress.  <br/>
 
 3.  For `baseline_vae`, refer to https://github.com/deepfindr/gvae.

## Run code
1. To run experiments in evaluation Table 1 (in main paper), <br/>
- Activate conda environment:
 `conda activate DiGNNExplainer` <br/>
- Navigate to `evaluation/main-paper/realistic_graphs/MMD` folder and run the following command for each of the datasets - `dblp`,`imdb`,`mutag`,`BA_shapes`,`Tree_Cycle`,`Tree_Grids`,`ba3`. For `DBLP` run

  ```
  python3 MMD_evaluation.py --dataset dblp
  ```
  
2. To run all jupyter notebooks, for evaluation Tables 2 and 3 in main paper (`evaluation/main-paper`), and experiments in supplementary material (`evaluation/supplementary`), 
- Activate conda environment:
 `conda activate DiGNNExplainer`

- Run jupyter notebook:
 `jupyter notebook`
 
- As an initial setup download the IMDB node features from [here](https://drive.google.com/file/d/1cYWwO4WgfafH3G0bOw69DQsLAFUiJs-5/view?usp=sharing) to `graph generator/diffusion models/sampled_features_diffusion/no_dependence/tabddpm/imdb`.<br/> 

3. For graph generation and node feature generation,<br/>
- To run `baseline_vae`:<br/>
	- Specify node feature size, dataset in `baseline_vae/config.py` as: 
	  ```
	  NODE_FEATURE_SIZE = 3066
	  DATASET = "imdb"   
	  ```
   	 	`NODE_FEATURE_SIZE` is the size of all features of the dataset.<br/>
      
	- Navigate to `graph generator/baseline_vae` folder and run
       	 `python3 train.py`
   
- To run `diffusion_graph_gen`:<br/>
	- Specify dataset in `graph generator/diffusion models/diffusion_graph_gen/configs/general/general_default.yaml`:
 
	  ```
	  dataset_name: 'dblp'
	  ```
	- Navigate to `diffusion_graph_gen` folder and run
	   `python3 main.py`

 - To run `DiTabDDPM` for DBLP Author class:<br/>
 	- Specify dataset in `configs/config.yaml`, and node class, node feature size in `graph generator/diffusion models/DiTabDDPM/configs/dataset/dblp.yaml`:
    
	  ```
	  dataset: dblp
	  node_class: 0
	  node_feature_size: 4
	  ```
	- Navigate to `DiTabDDPM` folder and run
          `python3 main.py`
          
 - To run original `TabDDPM` for DBLP Author class 0:<br/>
 	- Refer to [TabDDPM](https://github.com/yandex-research/tab-ddpm) for data preparation, and sample node features by running `pipeline.py` with the parameters:
    
	  ```
	  --config exp/author0/ddpm_cb_best/config.toml --train --sample
	  ```
          
     		
4. To run code in `baseline_explainers`,
- For `xgnn`, refer to [XGNN](https://github.com/divelab/DIG/tree/main/dig/xgraph/XGNN).<br/>
- For `gnninterpreter`, follow the installation steps of [GNNInterpreter](https://github.com/yolandalalala/GNNInterpreter/tree/main) and refer to [FACT2024_GNNInterpreter](https://github.com/MeneerTS/FACT2024_GNNInterpreter/tree/main/GNNInterpreter-Most-Recent-Version). <br/>
- For `d4explainer`, follow the installation steps of [D4Explainer](https://github.com/Graph-and-Geometric-Learning/D4Explainer/tree/main).<br/>

       
## Source of datasets
- DBLP : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html
- IMDB : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.IMDB.html
- MUTAG : https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.TUDataset.html
- BA-Shapes : https://www.dgl.ai/dgl_docs/generated/dgl.data.BAShapeDataset.html
- Tree-Cycle : https://www.dgl.ai/dgl_docs/generated/dgl.data.TreeCycleDataset.html
- Tree-Grids : https://www.dgl.ai/dgl_docs/generated/dgl.data.TreeGridDataset.html
- BA-3Motif : https://github.com/Wuyxin/ReFine/tree/main/data/BA3/raw


