import numpy as np
import scipy as sp
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor


class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = normalize_adj(subset_Mat)
        subset_sparse_tensor = sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()
        out_features = t.spmm(subset_sparse_tensor, subset_features)
        new_features = t.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(t.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features


class MLP(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu = nn.PReLU().cuda()
        x = prelu(x)
        # relu=nn.ReLU().cuda()
        # x = relu(x)
        # leaky=nn.LeakyReLU().cuda()
        # x = leaky(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x



