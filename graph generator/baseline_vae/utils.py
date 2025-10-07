from torch_geometric.utils import to_dense_adj
import torch
import torch.nn.functional as F
from scipy import spatial

from sklearn import preprocessing as pre
import numpy as np
from config import DEVICE as device

from config import (NODE_TYPES, EDGE_TYPES, NODE_FEATURE_SIZE)
NODE_FEATURE_SIZE = NODE_FEATURE_SIZE+len(NODE_TYPES)
def count_parameters(model):
    """
    Counts the number of parameters for a Pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_one_hot(x, options):
    """
    Converts a tensor of values to a one-hot vector
    based on the entries in options.
    """
    return torch.nn.functional.one_hot(x.long(), len(options))

def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd = logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div


def adj_recon_loss(adj_truth, adj_pred):
    return F.binary_cross_entropy(adj_truth, adj_pred)

def node_type_loss(truth, pred):
    return spatial.distance.cosine(truth, pred)

def rescale(x):
    x = x.reshape(-1, 1)
    x_norm = pre.MinMaxScaler().fit_transform(x)
    return x_norm

def node_feature_loss(original,sampled):
    #https://stackoverflow.com/questions/11405673/python-cosine-similarity-m-n-matrices
    A = np.array(rescale(original), dtype=object)
    B = np.array(rescale(sampled), dtype=object)

    Aflat = np.hstack(A)
    Bflat = np.hstack(B)

    return spatial.distance.cosine(Aflat, Bflat)


def triu_to_dense(triu_values, num_nodes):
    """
    Converts a triangular upper part of a matrix as flat vector
    to a squared adjacency matrix with a specific size (num_nodes).
    """
    dense_adj = torch.zeros((num_nodes, num_nodes)).to(device).float()
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
    dense_adj[triu_indices[0], triu_indices[1]] = triu_values
    dense_adj[tril_indices[0], tril_indices[1]] = triu_values
    return dense_adj

def get_adjacency(triu_logits,max_nodes):
    # Reshape triu predictions
    edge_matrix_shape = (int((max_nodes * (max_nodes - 1)) / 2), len(EDGE_TYPES) + 1)
    triu_preds_matrix = triu_logits.reshape(edge_matrix_shape)
    triu_preds = torch.argmax(triu_preds_matrix, dim=1)
    adjacency_matrix = triu_to_dense(triu_preds.float(), max_nodes)
    return adjacency_matrix

def get_nodetype(node_logits,max_nodes):
    node_matrix_shape = (max_nodes,len(NODE_TYPES))
    node_preds_matrix = node_logits.reshape(node_matrix_shape)
    node_preds = torch.argmax(node_preds_matrix[:, :len(NODE_TYPES)], dim=1)

    return node_preds
def slice_graph_targets(graph_id, batch_targets, node_targets,node_features, batch_index):
    """
    Slices out the upper triangular part of an adjacency matrix for
    a single graph from a large adjacency matrix for a full batch.
    --------
    graph_id: The ID of the graph (in the batch index) to slice
    batch_targets: A dense adjacency matrix for the whole batch
    batch_index: The node to graph map for the batch
    """
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph targets
    graph_targets = batch_targets[graph_mask][:, graph_mask]
    # Get triangular upper part of adjacency matrix for targets
    triu_indices = torch.triu_indices(graph_targets.shape[0], graph_targets.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()

    # Slice node targets
    graph_node_targets = node_targets[graph_mask]
    # Slice node features
    graph_node_feature_targets = node_features[graph_mask]
    return graph_targets[triu_mask],graph_node_targets,graph_node_feature_targets

def slice_graph_predictions(triu_logits, node_logits, feature_logits, graph_triu_size, start_point,graph_size, node_start_point,feature_size):
    """
    Slices out the corresponding section from a list of batch triu values.
    Given a start point and the size of a graph's triu, simply slices
    the section from the batch list.
    -------
    triu_logits: A batch of triu predictions of different graphs
    graph_triu_size: Size of the triu of the graph to slice
    start_point: Index of the first node of this graph
    """
    graph_logits_triu = torch.squeeze(
                    triu_logits[start_point:start_point + graph_triu_size]
                    )
    # Slice node logits
    graph_node_logits = torch.squeeze(
        node_logits[node_start_point:node_start_point + graph_size]
    )

    # Slice node logits
    graph_node_feature_logits = torch.squeeze(
        feature_logits[node_start_point:node_start_point + feature_size]
    )
    return graph_logits_triu, graph_node_logits, graph_node_feature_logits


def gvae_loss(MAX_NODES,triu_logits,node_logits,feature_logits, edge_index,node_types,node_features, mu, logvar, batch_index, kl_beta):
    """
    Calculates a weighted ELBO loss for a batch of graphs for the graph
    variational autoencoder model.
    """
    # Convert target edge index to dense adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))

    # Reconstruction loss per graph
    batch_recon_loss = []
    batch_node_type_loss = []
    batch_node_feature_loss = []
    batch_node_counter = 0
    graph_size = MAX_NODES * len(NODE_TYPES)
    graph_size_counter = 0
    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
        # Get upper triangular targets for this graph from the whole batch
        graph_targets_triu, node_targets, node_feature_targets = slice_graph_targets(graph_id,
                                                 batch_targets,
                                                 node_types,
                                                 node_features,
                                                 batch_index)

        # Get upper triangular predictions for this graph from the whole batch
        graph_predictions_triu, node_preds, node_feature_preds = slice_graph_predictions(triu_logits,
                                                         node_logits,feature_logits,
                                                         graph_targets_triu.shape[0],
                                                         batch_node_counter,
                                                         graph_size,
                                                         graph_size_counter,
                                                         MAX_NODES* NODE_FEATURE_SIZE)

        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_triu.shape[0]

        # Calculate edge-weighted binary cross entropy
        weight = graph_targets_triu.shape[0] / sum(graph_targets_triu)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss = bce(graph_predictions_triu.view(-1), graph_targets_triu.view(-1))
        batch_recon_loss.append(graph_recon_loss)

        node_type_pred = get_nodetype(node_preds, MAX_NODES)
        nodetype_loss = node_type_loss(node_targets, node_type_pred)
        batch_node_type_loss.append(nodetype_loss)

        feature_loss = node_feature_loss(node_feature_targets.float().detach().numpy(),
                                         node_feature_preds.reshape(MAX_NODES, NODE_FEATURE_SIZE).detach().numpy())
        batch_node_feature_loss.append(feature_loss)

        # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / num_graphs
    batch_node_type_loss = sum(batch_node_type_loss) / num_graphs
    batch_node_feature_loss = sum(batch_node_feature_loss) / num_graphs
    # KL Divergence
    kl_divergence = kl_loss(mu, logvar)

    return batch_recon_loss + batch_node_type_loss + batch_node_feature_loss + kl_beta * kl_divergence, kl_divergence








