import os

import numpy as np
import scipy
from scipy import sparse as sp
import torch as t
import torch.nn.functional as F
import torch.utils.data as dataloader
import datetime
import random
import sklearn

random.seed(123)


def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]  # (763, 6)

    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])  # [763x6]

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


# def knn_graph(disMat, k):
#     # 创建一个布尔矩阵，标记大于k的元素
#     mask = disMat > k  # 选择大于k的元素
#
#     # 获取符合条件的元素位置
#     row_index, col_index = np.nonzero(mask)
#
#     # 构建邻接矩阵
#     edges = np.array([row_index, col_index]).astype(int).T
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(disMat.shape[0], disMat.shape[0]),
#                         dtype=np.float32)
#     # adj = sklearn.preprocessing.normalize(adj, norm='l2', axis=1)
#
#     # 构建对称邻接矩阵
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     return adj


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
def ssl_loss(data1, data2, index, temp):
    index = t.unique(index)
    embeddings1 = data1[index]
    embeddings2 = data2[index]
    norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    pos_score = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim=1)
    all_score = t.mm(norm_embeddings1, norm_embeddings2.T)
    pos_score = t.exp(pos_score / temp)
    all_score = t.sum(t.exp(all_score / temp), dim=1)
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


# AdaDR
def common_loss(emb1, emb2):
    emb1 = emb1 - t.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - t.mean(emb2, dim=0, keepdim=True)
    emb1 = t.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = t.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = t.matmul(emb1, emb1.t())
    cov2 = t.matmul(emb2, emb2.t())
    cost = t.mean((cov1 - cov2) ** 2)
    return cost


# DRGCL
def semi(args, z1: t.Tensor, z2: t.Tensor):
    def sim(z1: t.Tensor, z2: t.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return t.mm(z1, z2.t())

    f = lambda x: t.exp(x / args.tau)
    refl_sim = f(args.intra * sim(z1, z1))  # torch.Size([663, 663])
    between_sim = f(args.inter * sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    return -t.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def semi_loss(args, z1: t.Tensor, z2: t.Tensor,
              mean: bool = True):
    def projection(args, z: t.Tensor) -> t.Tensor:
        fc1 = t.nn.Linear(args.hide_dim, 128).cuda()
        fc2 = t.nn.Linear(128, 256).cuda()
        fc3 = t.nn.Linear(256, args.hide_dim).cuda()
        z1 = F.elu(fc1(z))
        z2 = F.elu(fc2(z1))
        # z = t.sigmoid(fc1(z))
        return fc3(z2)

    h1 = projection(args, z1)
    h2 = projection(args, z2)
    l1 = semi(args, h1, h2, )
    l2 = semi(args, h2, h1, )
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


# ANSI escape codes for text color
class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def log(msg, color=None):
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if color:
        tem = f"{color}{tem}{Colors.RESET}"
    print(tem)


def sample_neighbors(adj, K):
    num_rows = adj.shape[0]
    sampled_neighbors = []

    # 筛选出 data 值为 1 的条目
    indices_ones = np.where(adj.data == 1)[0]

    # 获取对应的行和列索引
    row_indices = adj.row[indices_ones]
    col_indices = adj.col[indices_ones]
    # Create a list of neighbors for each row
    neighbors_list = [[] for _ in range(num_rows)]
    for row, col in zip(row_indices, col_indices):
        neighbors_list[row].append(col)

    # Sample neighbors
    for neighbors in neighbors_list:
        random.shuffle(neighbors)
        if len(neighbors) == 0:
            sampled_neighbors.append(t.tensor([-1] * K).long())  # No neighbors case
        elif len(neighbors) == K:
            sampled_neighbors.append(t.tensor(sorted(neighbors)).long())
        elif len(neighbors) > K:
            # sampled_neighbors.append(t.tensor(sorted(random.choices(neighbors, k=K))).long())
            sampled_neighbors.append(t.tensor(sorted(neighbors[:K])).long())
        else:
            _neighbors = neighbors
            _neighbors += random.choices(neighbors, k=K - len(neighbors))
            sampled_neighbors.append(t.tensor(sorted(_neighbors)).long())
    # Convert the list of neighbors to a tensor
    sampled_neighbors = t.stack(sampled_neighbors)
    return sampled_neighbors


# no itself
def _re_features(adj, features, target_features, K):
    # 传播之后的特征矩阵,size= (N, K, d )
    nodes_features = t.zeros(adj.shape[0], K, features.shape[1])
    sampled_neighbors_list = sample_neighbors(adj, K)  # (269 * K)
    for i, neighbors in enumerate(sampled_neighbors_list):
        if neighbors[0] != -1:
            # 将邻居特征放在后续列
            nodes_features[i, :, :] = features[neighbors]  # (K, d)
    return nodes_features.cuda()


def re_features(adj, features, target_features, K):
    # 传播之后的特征矩阵,size= (N, K+1, d )
    # neighbor_range = np.sum(adj.toarray(), axis=1)
    # min_neighbor, max_neighbor = neighbor_range.min, neighbor_range.max
    # if K > max_neighbor:
    #     K = max_neighbor
    nodes_features = t.zeros(target_features.shape[0], K + 1, features.shape[1])
    # 将目标特征放在第一列
    nodes_features[:, 0, :] = target_features
    sampled_neighbors_list = sample_neighbors(adj, K)  # (269 * K)
    for i, neighbors in enumerate(sampled_neighbors_list):
        if neighbors[0] != -1:
            # 将邻居特征放在后续列
            nodes_features[i, 1:, :] = features[neighbors]
    return nodes_features.cuda()


# no itselt
def _hops_features(homo_adj, hete_adj, features, target_features, hops):
    """

    :param homo_adj: (N, N)
    :param hete_adj: (N, M)
    :param features: (M, d)
    :param target_features: (N, d)
    :param hops: hop of drug to disease
    :return node_features: (N, hops+1, d)
    """
    homo_adj = sparse_mx_to_torch_sparse_tensor(homo_adj)
    hete_adj = sparse_mx_to_torch_sparse_tensor(hete_adj)
    features = features.float()
    target_features = target_features.float()
    nodes_features = t.zeros(target_features.shape[0], hops, features.shape[1])  # (N, K, d)
    # k-hop features
    for k in range(hops):
        if k == 0:
            x = t.matmul(hete_adj, features)  # (N, M) * (M, d) -> (N, d) : k-hop neighbors(disease) features of drug
        else:
            x = t.matmul(homo_adj, x)  # (N, N) * [ (N, M) * (M, d) ] -> (N, d)
        nodes_features[:, k, :] = x
    return nodes_features.cuda()


def hops_features(homo_adj, hete_adj, features, target_features, hops):
    """

    :param homo_adj: (N, N)
    :param hete_adj: (N, M)
    :param features: (M, d)
    :param target_features: (N, d)
    :param hops: hop of drug to disease
    :return node_features: (N, hops+1, d)
    """
    # rr_emb = t.nn.Embedding(663, 64)
    # dd_emb = t.nn.Embedding(409, 64)
    # features = rr_emb.float()
    # target_features = dd_emb.float()

    homo_adj = sparse_mx_to_torch_sparse_tensor(homo_adj)
    hete_adj = sparse_mx_to_torch_sparse_tensor(hete_adj)
    features = features.float()
    target_features = target_features.float()
    nodes_features = t.zeros(target_features.shape[0], hops + 1, features.shape[1])  # (N, K+1, d)
    # 0-hop features
    nodes_features[:, 0, :] = target_features  # (N, 1, d)
    # k-hop features
    for k in range(hops):
        if k == 0:
            x = t.matmul(hete_adj, features)  # (N, M) * (M, d) -> (N, d) : k-hop neighbors(disease) features of drug
        else:
            x = t.matmul(homo_adj, x)  # (N, N) * [ (N, M) * (M, d) ] -> (N, d)
        nodes_features[:, k + 1, :] = x
    return nodes_features.cuda()


def case_result(specific_name, drug_id, y_score):
    drug2score = {drug_id[i][0]: y_score[i] for i in range(len(y_score))}
    sorted_data = sorted(drug2score.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_potential_drug = {}
    for i in range(len(sorted_data)):
        sorted_potential_drug[sorted_data[i][0]] = sorted_data[i][1]

    path = os.path.join(".", "case_study/result/Fdataset/")
    if not os.path.exists(path):
        os.makedirs(path)
    potential_drug_file = "{}-potential-drug.txt".format(specific_name)
    with open(path + potential_drug_file, "w") as f:
        for key, value in sorted_potential_drug.items():
            f.write(str(key) + "\t" + "{}".format(value) + "\n")
