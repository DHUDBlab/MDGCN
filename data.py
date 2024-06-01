import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold

from args import args
from utils import knn_graph

_paths = {
    'Fdataset': './data/Fdataset/',
    'Cdataset': './data/Cdataset/',
    'Ldataset': './data/Ldataset/lagcn',
    'lrssl': './data/LRSSL',
}


class DataHandler(object):
    def __init__(self):
        self.num_neighbor = args.topK
        print("Starting processing ...")
        self.dir = _paths[args.dataset]
        if os.path.exists(os.path.join(self.dir, 'data.npy')):
            self.data = np.load(os.path.join(self.dir, 'data.npy'), allow_pickle=True)
        else:
            self.data = self.load_data(self.dir, args.dataset)  # trainMat, testMat, heteroMat
            np.save(os.path.join(self.dir, 'data.npy'), self.data)
        # self.data = self.load_data(self.dir, args.dataset)  # trainMat, testMat, heteroMat

    def load_data(self, file_path, data_name):
        association_matrix = None
        # GET association_matrix | disease_sim_feature | drug_sim_feature
        if data_name in ['Fdataset', 'Cdataset']:
            data = sio.loadmat(os.path.join(file_path, data_name + '.mat'))
            association_matrix = data['didr'].T
            self.dis_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.dis_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.dis_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values

        self.drug_num = association_matrix.shape[0]
        self.dis_num = association_matrix.shape[1]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)  # train:test=9:1
        pos_row, pos_col = np.nonzero(association_matrix)
        neg_row, neg_col = np.nonzero(1 - association_matrix)
        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        cv_num = 0
        cv_mat_data = {}
        self.cv_data = {}
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            # train data | pos:neg=9:9n
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = [0] * len(train_neg_edge[0])
            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_values = np.concatenate([train_pos_values, train_neg_values])
            trainMat = coo_matrix((train_values, (train_edge[0], train_edge[1])),
                                  shape=(self.drug_num, self.dis_num))
            # test data | pos:neg=1:n
            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = [1] * len(test_pos_edge[0])
            test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
            """
            # test | pos:neg=1:1 (AUPR will be highe, above 90%)
            test_neg_edge = np.stack([neg_row[test_neg_idx][0:len(test_pos_values)],
                                    neg_col[test_neg_idx][0:len(test_pos_values)]]) 
            """
            test_neg_values = [0] * len(test_neg_edge[0])
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_values = np.concatenate([test_pos_values, test_neg_values])
            testMat = coo_matrix((test_values, (test_edge[0], test_edge[1])),
                                 shape=(self.drug_num, self.dis_num))
            heteroMat = self.make_hetero_graph(trainMat)  # heterogeneous matrix
            cv_mat_data[cv_num] = [trainMat, testMat, heteroMat]
            cv_num += 1
        rrMat, ddMat = self.make_homo_graph()
        return cv_mat_data, rrMat, ddMat

    def make_hetero_graph(self, mat):
        # make ui adj
        a = sp.csr_matrix((self.drug_num, self.drug_num))
        b = sp.csr_matrix((self.dis_num, self.dis_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat.tocsr() != 0) * 1.0
        # mat = (mat != 0) * 1.0
        # mat = self.normalizeAdj(mat)

        # make cuda tensor
        # idxs = th.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        # vals = th.from_numpy(mat.data.astype(np.float32))
        # shape = th.Size(mat.shape)
        # return th.sparse.FloatTensor(idxs, vals, shape).cuda()
        return mat

    def make_homo_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        rr_adj = knn_graph(drug_sim, drug_num_neighbor)
        rr_adj = (rr_adj != 0) * 1.0
        # drug_graph = normalize(rr_adj + sp.eye(rr_adj.shape[0]))
        # drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        dis_sim = self.dis_sim_features
        dis_num_neighbor = self.num_neighbor
        if dis_num_neighbor > dis_sim.shape[0] or dis_num_neighbor < 0:
            dis_num_neighbor = dis_sim.shape[0]

        dd_adj = knn_graph(dis_sim, dis_num_neighbor)
        dd_adj = (dd_adj != 0) * 1.0
        # dis_graph = normalize(dd_adj + sp.eye(dd_adj.shape[0]))
        # dis_graph = sparse_mx_to_torch_sparse_tensor(dis_graph)

        # return drug_graph, dis_graph
        return rr_adj, dd_adj


if __name__ == '__main__':
    dataHandler = DataHandler(torch.device("cuda"))
    print(dataHandler)
