import networkx as nx
import pandas as pd
import torch_geometric as pyg
import numpy as np
import torch
import os.path as osp

from .base_graph_dataset import BaseGraphDataset
from .utils import default_ax, unpack_G

from torch_geometric.data import Data


class Ba3dataset(BaseGraphDataset):
    NODE_CLS = {
        0: 'blue',
        1: 'red',
        2: 'darkgreen',
        3: 'orange'
    }

    GRAPH_CLS = {
        0: 'House',
        1: 'Grid',
        2: 'Cycle'
    }

    def __init__(self, *,
                 name='BA-3motif',
                 url='none',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return ["BA-3motif.npy"]

    def download(self):
        if not osp.exists(osp.join(self.raw_dir,  "BA-3motif.npy")):
            print(
                "raw data of `BA-3motif.npy` doesn't exist, please download from https://github.com/Wuyxin/ReFine/tree/main/data/BA3/raw."
            )
            raise FileNotFoundError

    def generate(self):
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(
            osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True
        )

        data_list = []
        #alpha = 0.25
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(
            zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)
        ):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            #x = torch.zeros(node_idx.size(0), 4)
            #index = [i for i in range(node_idx.size(0))]
            #x[index, z] = 1
            #x = alpha * x + (1 - alpha) * torch.rand((node_idx.size(0), 4))
            feat = torch.tensor([1.0,0.0,0.0,0.0])
            x = feat.repeat(len(node_idx),1)
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            p = np.array(list(p.values()))

            data = Data(
                x=x,
                y=y,
                z=z,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=p,
                ground_truth_mask=ground_truth,
                name=f"BA-3motif{idx}",
                idx=idx,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        return data_list

    @default_ax
    def draw(self, G, pos=None, label=False, ax=None):
        # pos = pos or nx.kamada_kawai_layout(G)
        # nx.draw_networkx_nodes(G, pos,
        #                        ax=ax,
        #                        nodelist=G.nodes,
        #                        node_size=300,
        #                        edgecolors='black')
        #
        # nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=1, edge_color='tab:gray')
        nx.draw(G,node_size=100)

    def process(self):
        super().process()
