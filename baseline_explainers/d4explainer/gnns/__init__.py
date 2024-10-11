from .ba3motif_gnn import BA3MotifNet
from .mutag_gnn import Mutag_GCN
from .dblp_gnn import DBLP_GCN
from .imdb_gnn import IMDB_GCN
from .pubmed_gnn import PubMed_GCN
#Comment below lines for Syn_GCN_BAS Syn_GCN_TC Syn_GCN_TG in torch-geometric 2.4.0, as it throws an error
# from .BA_shapes_gnn import Syn_GCN_BAS
# from .tree_cycle_gnn import Syn_GCN_TC
# from .tree_grids_gnn import Syn_GCN_TG

__all__ = [
    "BA3MotifNet",
    "Mutag_GCN",
    "DBLP_GCN",
    "IMDB_GCN",
    "PubMed_GCN",
#Comment below lines for Syn_GCN_BAS Syn_GCN_TC Syn_GCN_TG in torch-geometric 2.4.0, as it throws an error
    # "Syn_GCN_BAS",
    # "Syn_GCN_TC",
    # "Syn_GCN_TG",

]
