import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from utils import knn_graph

script_dir = os.path.dirname(os.path.abspath(__file__))
_paths = {
    'Fdataset': script_dir + '/data/Fdataset/',
    'Cdataset': script_dir + '/data/Cdataset/',
    'Ldataset': script_dir + '/data/Ldataset/lagcn',
    'lrssl': script_dir + '/data/LRSSL',
}


class DataHandler(object):
    def __init__(self, args):
        self.topK = args.topK
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

            association_matrix = data.values  # (763, 681)

            self.dis_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values  # (681, 681)

            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values  # (763, 763)

        self.drug_num = association_matrix.shape[0]
        self.dis_num = association_matrix.shape[1]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)  # train:test=9:1
        pos_row, pos_col = np.nonzero(association_matrix)  # 取出非0元素的索引坐标，3051个

        neg_row, neg_col = np.nonzero(1 - association_matrix)  # 取出0元素的索引坐标，516552个

        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        cv_num = 0
        cv_mat_data = {}
        self.cv_data = {}
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
        # train data | pos:neg=9:9n
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])  # 随机取出2745个非0元素的坐标
            train_pos_values = [1] * len(train_pos_edge[0])  # 创建大小为2745的数组，值均为1

            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = [0] * len(train_neg_edge[0])

            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)  # 这里是全部数据9/10的坐标（2745个非0元素坐标在前）
            train_values = np.concatenate([train_pos_values, train_neg_values])  # 【1111,...,0000000],467643个

            # 创建稀疏矩阵，对于训练集没啥用，测试集可大大减小空间复杂度，最终形状都是(763, 681)
            trainMat = coo_matrix((train_values, (train_edge[0], train_edge[1])),
                                  shape=(self.drug_num, self.dis_num))  # 创建稀疏矩阵:467643个   (x,y) value

        # test data | pos:neg=1:n
            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = [1] * len(test_pos_edge[0])
            test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
            """
            # test | pos:neg=1:1 (AUPR will be higher, above 90%)
            test_neg_edge = np.stack([neg_row[test_neg_idx][0:len(test_pos_values)],
                                    neg_col[test_neg_idx][0:len(test_pos_values)]]) 
            """
            test_neg_values = [0] * len(test_neg_edge[0])
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_values = np.concatenate([test_pos_values, test_neg_values])
            testMat = coo_matrix((test_values, (test_edge[0], test_edge[1])),
                                 shape=(self.drug_num, self.dis_num))

        # 创建了个关于训练集的大稀疏矩阵（拼接了4个，）再转为稀疏矩阵
            heteroMat = self.make_hetero_graph(trainMat)  # (1444,1444)

            cv_mat_data[cv_num] = [trainMat, testMat, heteroMat]
            cv_num += 1

        # 邻居k  矩阵
        rrMat, ddMat = self.make_homo_graph()

        return cv_mat_data, rrMat, ddMat

    def make_hetero_graph(self, mat):
        # make ui adj
        a = sp.csr_matrix((self.drug_num, self.drug_num))
        b = sp.csr_matrix((self.dis_num, self.dis_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat.tocsr() != 0) * 1.0
        # rd_mat = mat[: self.drug_num, self.drug_num:]
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
        drug_sim = self.drug_sim_features  # (763x763)
        drug_num_neighbor = self.topK  # 一个数值

        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]
        rr_adj = knn_graph(drug_sim, drug_num_neighbor)
        rr_adj = (rr_adj != 0) * 1.0

        # rr_adj = (drug_sim - np.min(drug_sim)) / (np.max(drug_sim) - np.min(drug_sim))
        # rr_adj = coo_matrix(rr_adj)
        # rr_adj = rr_adj + rr_adj.T.multiply(rr_adj.T > rr_adj) - rr_adj.multiply(rr_adj.T > rr_adj)

        # disease feature graph
        dis_sim = self.dis_sim_features
        dis_num_neighbor = self.topK
        if dis_num_neighbor > dis_sim.shape[0] or dis_num_neighbor < 0:
            dis_num_neighbor = dis_sim.shape[0]

        dd_adj = knn_graph(dis_sim, dis_num_neighbor)
        dd_adj = (dd_adj != 0) * 1.0

        # dd_adj = (dis_sim - np.min(dis_sim)) / (np.max(dis_sim) - np.min(dis_sim))
        # dd_adj = coo_matrix(dd_adj)
        # dd_adj = dd_adj + dd_adj.T.multiply(dd_adj.T > dd_adj) - dd_adj.multiply(dd_adj.T > dd_adj)

        # return drug_graph, dis_graph
        return rr_adj, dd_adj
