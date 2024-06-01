import numpy as np
from scipy import sparse as sp
import torch as t
import torch.nn.functional as F
import torch.utils.data as dataloader
import datetime

from args import args


def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(int).T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(disMat.shape[0], disMat.shape[0]),
                        dtype=np.float32)
    # Remove diagonal elements
    # drug_adj = drug_adj - sp.dia_matrix((drug_adj.diagonal()[np.newaxis, :], [0]), shape=drug_adj.shape)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalizeAdj(mat):
    degree = np.array(mat.sum(axis=-1))
    epsilon = 1e-6
    degree[degree == 0] = epsilon
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # add mine start
    epsilon = 1e-6
    rowsum[rowsum == 0] = epsilon
    # add mine end
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sys_normalized_adjacency(adj):
    # adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a t sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data).float()
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)


def metaregular(em, _em, adj):
    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[:, t.randperm(embedding.shape[1])]
        corrupted_embedding = corrupted_embedding[t.randperm(embedding.shape[0])]
        return corrupted_embedding

    def score(x1, x2):
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        return t.sum(t.multiply(x1, x2), 1)

    drug_embeddings = _em
    Adj_Norm = t.from_numpy(np.sum(adj, axis=1)).float().cuda()
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    edge_embeddings = t.spmm(adj.cuda(), drug_embeddings) / Adj_Norm
    drug_embeddings = em
    graph = t.mean(edge_embeddings, 0)
    pos = score(drug_embeddings, graph)
    neg = score(row_column_shuffle(drug_embeddings), graph)
    global_loss = t.mean(-t.log(t.sigmoid(pos - neg)))
    return global_loss


# Contrastive Learning
def ssl_loss(data1, data2, index):
    index = t.unique(index)
    embeddings1 = data1[index]
    embeddings2 = data2[index]
    norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    pos_score = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim=1)
    all_score = t.mm(norm_embeddings1, norm_embeddings2.T)
    pos_score = t.exp(pos_score / args.ssl_temp)
    all_score = t.sum(t.exp(all_score / args.ssl_temp), dim=1)
    ssl_loss = (-t.sum(t.log(pos_score / all_score)) / (len(index)))
    return ssl_loss


class MyTransData(dataloader.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.data = coomat.data

    def __len__(self):
        return len(self.rows)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.data[idx]  # 返回行、列索引和真实标签


logmsg = ''
saveDefault = False


def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if save is not None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)
