import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from utils import knn_graph

_paths = {
    'Fdataset': '../data/Fdataset/',
    'Cdataset': '../data/Cdataset/',
    'Ldataset': '../data/Ldataset/lagcn',
    'lrssl': '../data/LRSSL',
}


class CaseDataHandler(object):
    def __init__(self, args):
        self.topK = args.topK
        self.all_case_id = args.all_case_id
        print("Starting processing ...")
        self.dir = _paths[args.dataset]
        # if os.path.exists(os.path.join(self.dir, 'data.npy')):
        #     self.data = np.load(os.path.join(self.dir, 'data.npy'), allow_pickle=True)
        # else:
        #     self.data = self.load_data(self.dir, args.dataset)  # trainMat, testMat, heteroMat
        #     np.save(os.path.join(self.dir, 'data.npy'), self.data)
        self.data = self.load_data(self.dir, args.dataset)  # trainMat, testMat, heteroMat

    def load_data(self, file_path, data_name):
        association_matrix = None
        # GET association_matrix | disease_sim_feature | drug_sim_feature
        data = sio.loadmat(os.path.join(file_path, data_name + '.mat'))
        association_matrix = data['didr'].T
        drug_id = data['Wrname']
        for i in range(len(drug_id)):
            drug_id[i] = drug_id[i][0]
        self.drug_id = drug_id
        self.dis_sim_features = data['disease']
        self.drug_sim_features = data['drug']
        self.drug_num = association_matrix.shape[0]
        self.dis_num = association_matrix.shape[1]

        case_num = 0
        case_mat_data = {}
        for case_id in self.all_case_id:
            train_matrix = association_matrix.copy()
            test_values = train_matrix[:, case_id].copy()
            train_matrix[:, case_id] = 0
            pos_row, pos_col = np.nonzero(train_matrix)
            neg_row, neg_col = np.nonzero(1 - train_matrix)
            assert len(pos_row) + len(neg_row) == np.prod(train_matrix.shape)

            train_pos_edge = np.stack([pos_row, pos_col])
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = np.stack([neg_row, neg_col])
            train_neg_values = [0] * len(train_neg_edge[0])
            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_values = np.concatenate([train_pos_values, train_neg_values])

            trainMat = coo_matrix((train_values, (train_edge[0], train_edge[1])),
                                  shape=(self.drug_num, self.dis_num))
            test_drug_id = [row for row in range(0, self.drug_num)]
            test_disease_id = [case_id] * len(test_values)
            testMat = coo_matrix((test_values, (test_drug_id, test_disease_id)),
                                 shape=(self.drug_num, self.dis_num))
            heteroMat = self.make_hetero_graph(trainMat)  # heterogeneous matrix
            case_mat_data[case_num] = [trainMat, testMat, heteroMat]
            case_num += 1
        rrMat, ddMat = self.make_homo_graph()
        return case_mat_data, rrMat, ddMat

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
        drug_num_neighbor = self.topK
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]
        rr_adj = knn_graph(drug_sim, drug_num_neighbor)
        rr_adj = (rr_adj != 0) * 1.0
        # drug_graph = normalize(rr_adj + sp.eye(rr_adj.shape[0]))
        # drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        dis_sim = self.dis_sim_features
        dis_num_neighbor = self.topK
        if dis_num_neighbor > dis_sim.shape[0] or dis_num_neighbor < 0:
            dis_num_neighbor = dis_sim.shape[0]

        dd_adj = knn_graph(dis_sim, dis_num_neighbor)
        dd_adj = (dd_adj != 0) * 1.0
        # dis_graph = normalize(dd_adj + sp.eye(dd_adj.shape[0]))
        # dis_graph = sparse_mx_to_torch_sparse_tensor(dis_graph)

        # return drug_graph, dis_graph
        return rr_adj, dd_adj


