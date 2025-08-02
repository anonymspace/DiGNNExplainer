import networkx as nx
import pandas as pd
import torch_geometric as pyg
import numpy as np
import torch
import os.path as osp
import os
import pathlib

from .base_graph_dataset import BaseGraphDataset
from .utils import default_ax, unpack_G


class TreeCycledataset(BaseGraphDataset):
    NODE_CLS = {
        0: 'blue',
        1: 'red',

    }

    GRAPH_CLS = {
        0: 'Single',
    }

    def __init__(self, *,
                 name='TreeCycle',
                 url='none',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)


    def generate(self):
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        file_path = os.path.join(base_path, 'data/TreeCycle/processed/data.pt')
        data_list = []
        data = torch.load(file_path)

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
