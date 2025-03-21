from gnninterpreter import *
import torch
from torch import nn
import random
import numpy as np
import time
from tqdm.auto import trange
import networkx as nx
import matplotlib.pyplot as plt
import os
import settings

NODE_FEATURES = 4
NUM_CLASSES = 4
MAX_NODES = 15
DATASET_NAME = 'BA_shapes'
CLASSES = [0, 1, 2, 3]

def test_interpreter(target_class, show=False):
    dataset = BAshapesdataset(seed=12345)
    batch_size = len(dataset.data.y)
    model = GCNClassifierNC(node_features=NODE_FEATURES,
                        num_classes=NUM_CLASSES,
                        hidden_channels=64,
                        num_layers=3)


    for seed in [15]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model.load_state_dict(torch.load('ckpts/bashapes_ckpt.pt'))

        mean_embeds = dataset.mean_embeddings_nc(DATASET_NAME ,model, batch_size=batch_size)

        print("="*30)
        print("="*30)
        print("Seed: {}".format(seed))
        print("="*30)     
        print("="*30)

        trainer = get_trainer(dataset,model,mean_embeds,seed)

        if target_class == 0:
            print("class 0")
            start_time = time.time()
            trainer[0].train(1000)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Time to train: {}".format(elapsed_time))
            #print(trainer[0].quantatitive())
            # if show:
            #     for i in range(1):
            #         trainer[0].evaluate(threshold=0.5, show=True, bernoulli=True, connected=True)
            expln_graph = trainer[0].evaluate(threshold=0.5, show=True,bernoulli=True,connected=True)
            print("="*30)

        elif target_class == 1:

            #class 1

            print("class 1")
            start_time = time.time()
            trainer[1].train(1000)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Time to train: {}".format(elapsed_time))
            #print(trainer[1].quantatitive())
            # if show:
            #     for i in range(1):
            #         trainer[1].evaluate(threshold=0.5, show=True, bernoulli=True, connected=True)
            expln_graph = trainer[1].evaluate(threshold=0.5, show=True,bernoulli=True,connected=True)
            print("=" * 30)

        elif target_class == 2:
            # class 2

            print("class 2")
            start_time = time.time()
            trainer[2].train(1000)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Time to train: {}".format(elapsed_time))
            #print(trainer[2].quantatitive())
            # if show:
            #     for i in range(1):
            #         trainer[2].evaluate(threshold=0.5, show=True, bernoulli=True, connected=True)
            expln_graph = trainer[2].evaluate(threshold=0.5, show=True, bernoulli=True, connected=True)
            print("=" * 30)

        elif target_class == 3:
            # class 3

            print("class 3")
            start_time = time.time()
            trainer[3].train(1000)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Time to train: {}".format(elapsed_time))
            #print(trainer[3].quantatitive())
            # if show:
            #     for i in range(1):
            #         trainer[3].evaluate(threshold=0.5, show=True, bernoulli=True, connected=True)
            expln_graph = trainer[3].evaluate(threshold=0.5, show=True, bernoulli=True, connected=True)
            print("=" * 30)


    return expln_graph

def get_trainer(dataset,model,mean_embeds,seed):
    trainer = {}
    sampler = {}

    #Trainer class 0 - Acyclic
    cls_idx = 0
    trainer[cls_idx] = TrainerNC(
        seed=seed,
        classes=CLASSES,
        sampler=(s := GraphSampler(DATASET_NAME,
            max_nodes=MAX_NODES,
            num_node_cls=len(dataset.NODE_CLS),
            temperature=0.2,
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),

            dict(key="logits", criterion=MeanPenalty(), weight=0),
            dict(key="omega", criterion=NormPenalty(order=1), weight=10),
            dict(key="omega", criterion=NormPenalty(order=1), weight=2),

            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=50),
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=5000, order=2, beta=1),
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10
    )

    # Trainer class 1 # house
    cls_idx = 1
    trainer[cls_idx] = TrainerNC(
        seed=seed,
        classes=CLASSES,
        sampler=(s := GraphSampler(DATASET_NAME,
            max_nodes=MAX_NODES,
            num_node_cls=len(dataset.NODE_CLS),
            temperature=0.2,
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1), #mu, cosine sim
            dict(key="logits", criterion=MeanPenalty(), weight=0),
            dict(key="omega", criterion=NormPenalty(order=1), weight=1),   #L1
            dict(key="omega", criterion=NormPenalty(order=2), weight=1),    #L2

            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=50),  #Rc
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=5000, order=2, beta=1),           #Rb
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10                                                        #K
    )


    # Trainer class 2 #house
    cls_idx = 2
    trainer[cls_idx] = TrainerNC(
        seed=seed,
        classes=CLASSES,
        sampler=(s := GraphSampler(DATASET_NAME,
            max_nodes=MAX_NODES,
            num_node_cls=len(dataset.NODE_CLS),
            temperature=0.2,
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),

            dict(key="logits", criterion=MeanPenalty(), weight=0),
            dict(key="omega", criterion=NormPenalty(order=1), weight=1),
            dict(key="omega", criterion=NormPenalty(order=2), weight=1),

            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=50),
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=5000, order=2, beta=1),
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10
    )

    # Trainer class 3 #house
    cls_idx = 3
    trainer[cls_idx] = TrainerNC(
        seed=seed,
        classes=CLASSES,
        sampler=(s := GraphSampler(DATASET_NAME,
            max_nodes=MAX_NODES,
            num_node_cls=len(dataset.NODE_CLS),
            temperature=0.2,
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=1),

            dict(key="logits", criterion=MeanPenalty(), weight=0),
            dict(key="omega", criterion=NormPenalty(order=1), weight=1),
            dict(key="omega", criterion=NormPenalty(order=2), weight=1),

            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=50),

        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=dataset,
        budget_penalty=BudgetPenalty(budget=5000, order=2, beta=1),
        target_probs={cls_idx: (0.9, 1)},
        k_samples=10
    )
    return trainer


def get_avg_fidelity(graph_list):
    class_avg_fidelity = []
    for i, expln_graph in enumerate(graph_list):

        fid_score_list = []

        motifs_path = '../../evaluation/motifs/BAshapes/'

        files_motif = os.listdir(motifs_path)

        for index_m, file_m in enumerate(files_motif):
            filepath_m = os.path.join(motifs_path, file_m)

            motif_graph = nx.read_gexf(filepath_m)

            GM = nx.algorithms.isomorphism.GraphMatcher(expln_graph, motif_graph)
            x = 1 if GM.subgraph_is_isomorphic() else 0

            fid_score_list.append(x)

        class_avg_fidelity.append(np.mean(fid_score_list))

    return np.mean(class_avg_fidelity)
if __name__ == "__main__":

    mean_fidelity_list = []
    avg_prob_list = []
    for nodesize in range(10, 16):
        MAX_NODES = nodesize
        print('MAX_NODES', MAX_NODES)

        expln_graphs_list = []

        for i in range(0, 10):
            print('Run' + str(i))
            expln_graphs = []
            settings.class_prob_list = []
            # start time
            start = time.time()
            for target_class in CLASSES:
                G = test_interpreter(target_class, show=True)

                expln_graphs.append(G)

                # nx.draw(G, node_size=100)
                # plt.savefig(DATASET_NAME + str(target_class) + '.pdf')
                # plt.show()
                # plt.clf()

            # end time
            duration = time.time() - start
            print('Total Time for all classes(in secs)', duration)

            avg_prob_list.append(np.mean(settings.class_prob_list))
            expln_graphs_list.append(expln_graphs)

        avg_fidelity_list = []

        for i in range(0, 10):
            avg_fidelity = get_avg_fidelity(expln_graphs_list[i])

            print('Run' + str(i), avg_fidelity)
            avg_fidelity_list.append(avg_fidelity)
        # print(np.mean(avg_fidelity_list))
        mean_fidelity_list.append(np.mean(avg_fidelity_list))

    print('Average fidelity for 10-15 node sizes', np.mean(mean_fidelity_list))
    print('Average probability for 10-15 node sizes', np.mean(avg_prob_list))
