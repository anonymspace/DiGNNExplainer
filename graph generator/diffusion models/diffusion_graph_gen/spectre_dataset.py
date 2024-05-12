import os
import pathlib
import torch
import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import csv
from littleballoffur import ForestFireSampler
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# def remap_indices(G):
#     val_list = [*range(0, G.number_of_nodes(), 1)]
#     return dict(zip(G,val_list))


class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, node_size,transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.node_size = node_size
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        """
                """
#
        file_path = '../real_pubmed/'+str(self.node_size)+'.pt'


        adjs, node_types = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append([adj,node_types[i]])
            elif i in val_indices:
                val_data.append([adj,node_types[i]])
            elif i in test_indices:
                test_data.append([adj,node_types[i]])
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []


        for data in raw_dataset:
            adj, node_types = data
            n = adj.shape[-1]
            # For DBLP and PubMed
            node_classes = [0, 1, 2, 3]
            #For IMDB
            #node_classes = [0, 1, 2]
            #node_types = node_type

            X = F.one_hot(torch.tensor(node_types), num_classes=len(node_classes)).float()
            y = torch.zeros([1, 0]).float()

            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)

            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            #data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])



class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg,size):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, node_size = size),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path,node_size = size),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path,node_size = size)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        # For DBLP and PubMed
        self.node_types = torch.tensor([0,1,2,3])
        #For IMDB
        #self.node_types = torch.tensor([0, 1, 2])
        self.edge_types = self.datamodule.edge_counts()


        super().complete_infos(self.n_nodes, self.node_types)

