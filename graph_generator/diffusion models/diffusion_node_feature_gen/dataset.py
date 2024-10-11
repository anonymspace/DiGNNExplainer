import os
import pathlib
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


import torch_geometric.transforms as T
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.datasets import IMDB
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class FeatureDataset(InMemoryDataset):
    def __init__(self, dataset, split, root,transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset.name
        self.node_class = dataset.node_class
        self.node_feature_size = dataset.node_feature_size
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        # """
        # Download dataset
        # """

        train_data = []
        val_data = []
        test_data = []
        #class0 = pd.DataFrame()
        imp_feat = pd.DataFrame()

        if self.dataset_name == 'dblp':
            print('Dataset name',self.dataset_name)
            dataset = DBLP(root='./dblp_data', transform=T.Constant(node_types='conference'))
            data = dataset[0]
            #Author--------------------------------------------------
            author = data['author'].x.tolist()
            df = pd.DataFrame(author)
            df['class'] = data['author'].y.tolist()
            # Feature selection for Author class 0
            class0 = df[df['class'] == self.node_class].drop(['class'], axis=1)
            X = class0
            #imp_feat = class0

            #--------------------------------------------------------------------------------------------------
            # col_sum = X.sum(axis=0)
            # #sorted_colsum = sorted(col_sum, reverse=True)
            # colsum_df = pd.DataFrame(col_sum)
            #
            # sorted_colsum = sorted(col_sum, reverse=True)[:50]
            # index_list = list(np.ravel(colsum_df[colsum_df[0].isin(sorted_colsum)].index))
            #
            # imp_feat = X[index_list]

            #counting 1s----------------------------------------------------------------------------------------------------
            #
            col_sum = X.sum(axis=0)
            sorted_colsum = sorted(col_sum, reverse=True)
            index_list = []
            for i in sorted_colsum[:self.node_feature_size]:
                index_list.append(list(col_sum).index(i))
            imp_feat = X[index_list]
            # # ------------------------------------------------------------------------------------------------------------------
            # #Paper----------------------------------------------------
            # paper = data['paper'].x.tolist()
            # df_paper = pd.DataFrame(paper)
            # # # #imp_feat = df_paper
            # # #
            # X = df_paper
            # col_sum = X.sum(axis=0)
            # sorted_colsum = sorted(col_sum, reverse=True)
            # index_list = []
            # for i in sorted_colsum[:8]:
            #     index_list.append(list(col_sum).index(i))
            # imp_feat = X[index_list]

            #----------------------------------------------------------------

        elif self.dataset_name == 'imdb':
            dataset = IMDB(root='./imdb_data')
            data = dataset[0]
            movie = data['movie'].x.tolist()
            df = pd.DataFrame(movie)
            df['class'] = data['movie'].y.tolist()
            class0 = df[df['class'] == self.node_class].drop(['class'], axis=1)

            # -----------------------------------count 1s logic-----------------------
            #selects top k features with most number of 1s
            col_sum = class0.sum(axis=0)
            colsum_df = pd.DataFrame(col_sum)

            sorted_colsum = sorted(col_sum, reverse=True)[:self.node_feature_size]
            index_list = list(np.ravel(colsum_df[colsum_df[0].isin(sorted_colsum)].index))

            imp_feat = class0[index_list]



        imp_feat = imp_feat.drop(imp_feat[imp_feat.sum(axis=1)==0.0].index)
            # Using train/test/val -80/10/10
            # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
        train, val, test = np.split(imp_feat.sample(frac=1, random_state=42),
                                        [int(.8 * len(imp_feat)), int(.9 * len(imp_feat))])


        train_data.append(torch.tensor(train.values))
        val_data.append(torch.tensor(val.values))
        test_data.append(torch.tensor(test.values))

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])



    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for d in raw_dataset:

            node_feature = torch.tensor(d)
            data = torch_geometric.data.Data(feature=node_feature)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class FeatureDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': FeatureDataset(dataset=self.cfg.dataset,
                                                 split='train', root=root_path),
                    'val': FeatureDataset(dataset=self.cfg.dataset,
                                        split='val', root=root_path),
                    'test': FeatureDataset(dataset=self.cfg.dataset,
                                        split='test', root=root_path)}


        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class FeatureDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.feature_types = self.datamodule.feature_counts()


