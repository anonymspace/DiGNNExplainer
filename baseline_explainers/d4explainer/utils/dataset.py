import os

from datasets import  BA3Motif, Mutag, SynGraphDataset
from datasets.dblp_dataset import DBLPDataset
from datasets.imdb_dataset import IMDBDataset
#from datasets.pubmed_dataset import PubMedDataset


def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
    if name == "mutag":
        folder = os.path.join(root, "MUTAG")
        train_dataset = Mutag(folder, mode="training")
        test_dataset = Mutag(folder, mode="testing")
        val_dataset = Mutag(folder, mode="evaluation")

    elif name == "ba3":
        folder = os.path.join(root, "BA3")
        train_dataset = BA3Motif(folder, mode="training")
        test_dataset = BA3Motif(folder, mode="testing")
        val_dataset = BA3Motif(folder, mode="evaluation")
    elif name == "BA_shapes":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="BA_shapes")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="BA_shapes")
        train_dataset = SynGraphDataset(folder, mode="training", name="BA_shapes")
    elif name == "dblp":
        folder = os.path.join(root)
        test_dataset = DBLPDataset(folder, mode="testing", name="dblp")
        val_dataset = DBLPDataset(folder, mode="evaluating", name="dblp")
        train_dataset = DBLPDataset(folder, mode="training", name="dblp")
    elif name == "imdb":
        folder = os.path.join(root)
        test_dataset = IMDBDataset(folder, mode="testing", name="imdb")
        val_dataset = IMDBDataset(folder, mode="evaluating", name="imdb")
        train_dataset = IMDBDataset(folder, mode="training", name="imdb")
    # elif name == "pubmed":
    #     folder = os.path.join(root)
    #     test_dataset = PubMedDataset(folder, mode="testing", name="pubmed")
    #     val_dataset = PubMedDataset(folder, mode="evaluating", name="pubmed")
    #     train_dataset = PubMedDataset(folder, mode="training", name="pubmed")
    elif name == "Tree_Cycle":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="Tree_Cycle")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="Tree_Cycle")
        train_dataset = SynGraphDataset(folder, mode="training", name="Tree_Cycle")
    elif name == "Tree_Grids":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="Tree_Grids")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="Tree_Grids")
        train_dataset = SynGraphDataset(folder, mode="training", name="Tree_Grids")


    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset
