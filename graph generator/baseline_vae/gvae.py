import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv

from torch_geometric.nn import BatchNorm
import utils
from tqdm import tqdm
import networkx as nx
import os
import random
import time

from config import (NODE_TYPES, EDGE_TYPES, NODE_FEATURE_SIZE, DATASET)

class GraphVAE(nn.Module):
    def __init__(self,MAX_NODES):
        super(GraphVAE, self).__init__()
        self.encoder_embedding_size = MAX_NODES
        #self.edge_dim = 1
        self.latent_embedding_size = 64
        self.num_edge_types = len(EDGE_TYPES)
        self.num_node_types = len(NODE_TYPES)
        self.max_nodes = MAX_NODES
        self.node_feature_size = NODE_FEATURE_SIZE +len(NODE_TYPES)
        #self.node_feature_size = NODE_FEATURE_SIZE
        self.decoder_hidden_neurons = 512
        # Encoder layers
        self.conv1 = TransformerConv(self.node_feature_size,
                                    self.encoder_embedding_size,
                                    heads=4,
                                    concat=False,
                                    beta=True,

                                     )
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.conv2 = TransformerConv(self.encoder_embedding_size,
                                    self.encoder_embedding_size,
                                    heads=4,
                                    concat=False,
                                    beta=True,

                                     )
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.conv3 = TransformerConv(self.encoder_embedding_size,
                                    self.encoder_embedding_size,
                                    heads=4,
                                    concat=False,
                                    beta=True,

                                     )
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        self.conv4 = TransformerConv(self.encoder_embedding_size,
                                    self.encoder_embedding_size,
                                    heads=4,
                                    concat=False,
                                    beta=True,

                                     )



        # Linear layer1   # Mu
        # Linear Layer2  # Log var
        self.mu_transform = Linear(self.encoder_embedding_size,
                                            self.latent_embedding_size)
        self.logvar_transform = Linear(self.encoder_embedding_size,
                                            self.latent_embedding_size)

        # Decoder layers
        # Linear 1
        # Linear 2
        #Linear Layer3 #Decode nodes
        # Linear Layer4 #Decode edges

        self.linear_1 = Linear(self.latent_embedding_size, self.decoder_hidden_neurons)
        self.linear_2 = Linear(self.decoder_hidden_neurons, self.decoder_hidden_neurons)

        # --- Node decoding (outputs a matrix: (max_num_nodes * #node_types)
        node_output_dim = self.max_nodes*self.num_node_types
        self.node_decode = Linear(self.decoder_hidden_neurons, node_output_dim)

        # --- Edge decoding (outputs a triu tensor: (max_num_nodes*(max_num_nodes-1)/2*(#edge_types + 1) ))
        edge_output_dim = int(((self.max_nodes * (self.max_nodes - 1)) / 2) * (self.num_edge_types + 1))
        self.edge_decode = Linear(self.decoder_hidden_neurons, edge_output_dim)

        node_feature_dim = self.max_nodes*self.node_feature_size
        self.node_feature_decode = Linear(self.decoder_hidden_neurons, node_feature_dim)


    def encode(self, x, edge_index, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index).relu()
        x = self.bn3(x)
        x = self.conv4(x, edge_index).relu()



        # Latent transform layers
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)
        return mu, logvar

    def decode_graph(self, graph_z):
        """
        Decodes a latent vector into a continuous graph representation
        consisting of node types and edge types.
        """
        # Pass through shared layers
        z = self.linear_1(graph_z).relu()
        z = self.linear_2(z).relu()
        # Decode node types
        node_logits = self.node_decode(z)
        # Decode edge types
        edge_logits = self.edge_decode(z)
        # Decode edge types
        node_feature_logits = self.node_feature_decode(z)

        return node_logits, edge_logits, node_feature_logits

    def decode(self, z, batch_index):
        node_logits = []
        triu_logits = []
        feature_logits = []

        # Iterate over graphs in batch
        for graph_id in torch.unique(batch_index):
            # Get latent vector for this graph
            graph_z = z[graph_id]

            # Recover graph from latent vector
            node_type_logits, edge_logits, node_feature_logits = self.decode_graph(graph_z)

            # Store per graph results
            node_logits.append(node_type_logits)
            triu_logits.append(edge_logits)
            feature_logits.append(node_feature_logits)

        # Concatenate all outputs of the batch
        node_logits = torch.cat(node_logits)
        triu_logits = torch.cat(triu_logits)
        feature_logits = torch.cat(feature_logits)
        return triu_logits, node_logits, feature_logits

    def reparameterize(self, mu, logvar):
        if self.training:
            # Get standard deviation
            std = torch.exp(logvar)
            # Returns random numbers from a normal distribution
            eps = torch.randn_like(std)
            # Return sampled values
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index, batch_index):

        # Encode the graph
        mu, logvar = self.encode(x, edge_index, batch_index)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode latent vector into original graph
        triu_logits, node_logits, feature_logits= self.decode(z, batch_index)

        return triu_logits, node_logits, feature_logits,mu, logvar

    def save_syn_graphs(self, graph, node_feature, classes, max_num_nodes, filename):
        graph_path = "sampled_graphs_vae/" + DATASET + '/' + filename
        os.makedirs(graph_path)

        #save graph
        nx.write_gexf(graph, graph_path + '/' + filename + "_G" + str(max_num_nodes) + ".gexf")
        # save node feature
        torch.save(torch.tensor(node_feature), graph_path + '/' + filename + "_node_features" + str(max_num_nodes) + ".pt")
        # save node classes
        torch.save(torch.tensor(classes), graph_path + '/' + filename + "_node_class" + str(max_num_nodes) + ".pt")

    def sample_graphs(self, num=100):
        for _ in tqdm(range(num)):
            z = torch.randn(1, self.latent_embedding_size)
            # Get model output (this could also be batched)
            dummy_batch_index = torch.Tensor([0]).int()
            triu_logits, node_logits, feature_logits= self.decode(z, dummy_batch_index)

            # Reshape triu predictions
            edge_matrix_shape = (int((self.max_nodes * (self.max_nodes - 1)) / 2), self.num_edge_types + 1)
            triu_preds_matrix = triu_logits.reshape(edge_matrix_shape)
            triu_preds = torch.argmax(triu_preds_matrix, dim=1)

            # Reshape node predictions
            node_matrix_shape = (self.max_nodes, self.num_node_types)
            node_preds_matrix = node_logits.reshape(node_matrix_shape)
            node_preds = torch.argmax(node_preds_matrix[:, :self.num_node_types], dim=1)

            #Saving graph, node type, node feature
            node_types = node_preds.type(torch.int64)

            node_features = feature_logits.reshape(self.max_nodes,self.node_feature_size)

            adjacency_matrix = utils.triu_to_dense(triu_preds.float(), self.max_nodes)


            graph = nx.from_numpy_array(adjacency_matrix.numpy())

            nx.set_node_attributes(graph, dict(zip(graph.nodes(), list(node_types.numpy()))),
                                   'type')
            nx.set_node_attributes(graph, dict(zip(graph.nodes(), node_features.detach().numpy())),
                                   'feature')

            edges_to_remove = random.sample(list(graph.edges), round(graph.number_of_edges()*0.6))
            graph.remove_edges_from(edges_to_remove)

            graph_node_types = list(nx.get_node_attributes(graph, "type").values())

            graph_node_features = list(nx.get_node_attributes(graph, "feature").values())

            for n in graph.nodes:
                del graph.nodes[n]['type']
                del graph.nodes[n]['feature']

            self.save_syn_graphs(graph, graph_node_features, graph_node_types, graph.number_of_nodes(),
                            "sync" + str(random.randint(1, 1000000)) + time.strftime("%H%M%S"))




