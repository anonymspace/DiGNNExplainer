import os
import numpy as np
import networkx as nx
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--dataset", type=str, default="mutag")
    parser.add_argument("--nodes", type=int, default=11)
    return parser.parse_args()

args = parse_args()

if args.dataset == 'dblp':
    from dblp_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/dblp_10to15/'
    CLASSES = [0, 1, 2, 3]

if args.dataset == 'imdb':
    from imdb_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/imdb_5to10/'
    CLASSES = [0, 1, 2]

if args.dataset == 'pubmed':
    from pubmed_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/pubmed_10to15/'
    CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

if args.dataset == 'mutag':
    from mutag_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/motif_mutag/'
    CLASSES = [0, 1]

if args.dataset == 'BA_shapes':
    from BA_shapes_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/BAshapes_10to15/'
    CLASSES = [0, 1, 2, 3]

if args.dataset == 'Tree_Cycle':
    from Tree_Cycle_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/TreeCycle_10to15/'
    CLASSES = [0, 1]

if args.dataset == 'Tree_Grids':
    from Tree_Grids_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/TreeGrids_10to15/'
    CLASSES = [0, 1]

if args.dataset == 'ba3':
    from ba3_gnn_explain import gnn_explain
    motifs_path = '../motifs_real/motif_ba3/'
    CLASSES = [0, 1, 2]

def get_avg_fidelity(graph_list):
    class_avg_fidelity = []
    for i, A in enumerate(graph_list):

        expln_graph = nx.from_numpy_array(A)
        fid_score_list = []
        if args.dataset == 'ba3':
            if i == 0:
                path = motifs_path + 'motif_ba3_class0/'
            elif i == 1:
                path = motifs_path + 'motif_ba3_class1/'
            elif i == 2:
                path = motifs_path + 'motif_ba3_class2/'
        else:
            path = motifs_path

        files_motif = os.listdir(path)

        for index_m, file_m in enumerate(files_motif):
            filepath_m = os.path.join(path, file_m)

            motif_graph = nx.read_gexf(filepath_m)

            GM = nx.algorithms.isomorphism.GraphMatcher(expln_graph, motif_graph)
            x = 1 if GM.subgraph_is_isomorphic() else 0

            fid_score_list.append(x)

        class_avg_fidelity.append(np.mean(fid_score_list))

    return np.mean(class_avg_fidelity)

expln_graphs_list = []
for i in range(0, 10):
    print('Run'+str(i))
    expln_graphs = []
    for target_class in CLASSES:

        explainer = gnn_explain(args.nodes, 30,  target_class, 50)  ####arguments: (max_node, max_step, target_class, max_iters)
        adj, prob = explainer.train()

        expln_graphs.append(adj.detach().cpu().numpy())
        print(target_class, adj, prob)
    expln_graphs_list.append(expln_graphs)


avg_fidelity_list = []

for i in range(0,10):

    avg_fidelity = get_avg_fidelity(expln_graphs_list[i])

    print('Run'+str(i),avg_fidelity)
    avg_fidelity_list.append(avg_fidelity)
print(np.mean(avg_fidelity_list))



