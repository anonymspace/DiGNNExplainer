import networkx as nx
import pandas as pd
import torch_geometric as pyg

from .base_graph_dataset import BaseGraphDataset
from .utils import default_ax, unpack_G


class MUTAGDataset(BaseGraphDataset):
    NODE_CLS = {
        0: 'C',
        1: 'N',
        2: 'O',
        3: 'F',
        4: 'I',
        5: 'Cl',
        6: 'Br',
    }



    GRAPH_CLS = {
        0: 'nonmutagen',
        1: 'mutagen',
    }

    EDGE_CLS = {
        0: 'aromatic',
        1: 'single',
        2: 'double',
        3: 'triple',
    }

 

    def __init__(self, *,
                 name='MUTAG',
                 url="https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip",
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return ["MUTAG/MUTAG_A.txt",
                "MUTAG/MUTAG_graph_indicator.txt",
                "MUTAG/MUTAG_graph_labels.txt",
                "MUTAG/MUTAG_edge_labels.txt",
                "MUTAG/MUTAG_node_labels.txt"]

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_zip(f'{self.raw_dir}/MUTAG.zip', self.raw_dir)

    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        graph_labels = pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) > 0
        edge_labels = pd.read_csv(self.raw_paths[3], header=None)[0].to_numpy(dtype=int)
        node_labels = pd.read_csv(self.raw_paths[4], header=None)[0].to_numpy(dtype=int)
        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, dict(enumerate(node_labels)), name='label')
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name='graph')
        nx.set_edge_attributes(super_G, dict(zip(zip(*edges.T), edge_labels)), name='label')
        return unpack_G(super_G)


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
