import torch as t
import numpy as np

from layers import MLP, GCN_layer, Attention
from utils import metaregular, re_features

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        z = torch.mean(x, dim=0)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z


class MODEL(nn.Module):
    def __init__(self, args, drug_num, dis_num, rrMat, ddMat, rdMat, hide_dim, layer_num):
        super(MODEL, self).__init__()
        self.args = args
        self.drug_num = drug_num
        self.dis_num = dis_num
        self.rrMat = rrMat
        self.ddMat = ddMat
        self.rdMat = rdMat
        self.hide_dim = hide_dim
        self.layer_num = layer_num

        self.encoder = nn.ModuleList()
        for i in range(0, self.layer_num):
            self.encoder.append(GCN_layer())

        self.gating_weight_rb = nn.Parameter(t.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weight_rb.data)
        self.gating_weight_r = nn.Parameter(t.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weight_r.data)
        self.gating_weight_db = nn.Parameter(t.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weight_db.data)
        self.gating_weight_d = nn.Parameter(t.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weight_d.data)

        self.k = self.args.rank
        k = self.k
        self.mlp_r1 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp_r2 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp_d1 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp_d2 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.meta_net_r = nn.Linear(hide_dim * 3, hide_dim, bias=True)
        self.meta_net_d = nn.Linear(hide_dim * 3, hide_dim, bias=True)

        self.embedding_dict = nn.ModuleDict({
            'rr_emb': t.nn.Embedding(drug_num, hide_dim).cuda(),
            'dd_emb': t.nn.Embedding(dis_num, hide_dim).cuda(),
            'drug_emb': t.nn.Embedding(drug_num, hide_dim).cuda(),
            'dis_emb': t.nn.Embedding(dis_num, hide_dim).cuda(),
        })

        # 在模型初始化时添加SE模块
        self.se_rd = SEBlock(in_channels=64, reduction=16)

    def self_gating_r(self, em):
        return t.multiply(em, t.sigmoid(t.matmul(em, self.gating_weight_r) + self.gating_weight_rb))

    def self_gating_d(self, em):
        return t.multiply(em, t.sigmoid(t.matmul(em, self.gating_weight_d) + self.gating_weight_db))


    def forward(self, ifTraining, uid, iid, norm=1):
        dis_index = np.arange(0, self.dis_num)
        drug_index = np.arange(0, self.drug_num)
        rd_index = np.array(drug_index.tolist() + [i + self.drug_num for i in dis_index])

        # Initialize Embeddings
        drug_embed0 = self.embedding_dict['drug_emb'].weight
        dis_embed0 = self.embedding_dict['dis_emb'].weight
        # drug Emb, dis Emb, drug-dis Emb
        rr_embed0 = self.self_gating_r(drug_embed0)
        dd_embed0 = self.self_gating_d(dis_embed0)
        rd_embeddings = t.cat([drug_embed0, dis_embed0], 0)

        all_drug_embeddings = [rr_embed0]
        all_dis_embeddings = [dd_embed0]
        all_rd_embeddings = [rd_embeddings]

        # Encoder
        for i in range(len(self.encoder)):
            gcn = self.encoder[i]
            if i == 0:
                # first layer output
                drugEmbeddings0 = gcn(rr_embed0, self.rrMat, drug_index)
                disEmbeddings0 = gcn(dd_embed0, self.ddMat, dis_index)
                rdEmbeddings0 = gcn(rd_embeddings, self.rdMat, rd_index)

            else:
                # next layer output
                drugEmbeddings0 = gcn(drugEmbeddings, self.rrMat, drug_index)
                disEmbeddings0 = gcn(disEmbeddings, self.ddMat, dis_index)
                rdEmbeddings0 = gcn(rdEmbeddings, self.rdMat, rd_index)

            rd_random_noise = t.rand_like(rdEmbeddings0).cuda()
            rdEmbeddings0 += t.sign(rdEmbeddings0) * F.normalize(rd_random_noise, dim=-1) * self.args.eps
            rd_drugEmbedding0, rd_disEmbedding0 = t.split(rdEmbeddings0, [self.drug_num, self.dis_num])
            drugEd = (0.5 * drugEmbeddings0 + 0.5 * rd_drugEmbedding0)
            disEd = (0.5 * disEmbeddings0 + 0.5 * rd_disEmbedding0)

            drugEmbeddings = drugEd
            disEmbeddings = disEd
            rdEmbeddings = t.cat([drugEd, disEd], 0)
            # record output
            if norm == 1:
                norm_embeddings = F.normalize(drugEmbeddings0, p=2, dim=1)
                all_drug_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(disEmbeddings0, p=2, dim=1)
                all_dis_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(rdEmbeddings0, p=2, dim=1)
                all_rd_embeddings += [norm_embeddings]
            else:
                all_drug_embeddings += [drugEmbeddings]
                all_dis_embeddings += [disEmbeddings]
                all_rd_embeddings += [rdEmbeddings]

        drugEmbedding1 = t.stack(all_drug_embeddings, dim=1)
        drugEmbedding = t.mean(drugEmbedding1[:, :self.args.layers+1], dim=1)

        disEmbedding1 = t.stack(all_dis_embeddings, dim=1)
        disEmbedding = t.mean(disEmbedding1[:, :self.args.layers+1], dim=1)

        rdEmbedding = t.stack(all_rd_embeddings, dim=1)
        rdEmbedding = t.mean(rdEmbedding[:, :self.args.layers-2], dim=1)

        rd_drugEmbedding, rd_disEmbedding = t.split(rdEmbedding, [self.drug_num, self.dis_num])

        # Regularization: the constraint of transformed reasonableness
        meta_reg_loss = 0
        if ifTraining:
            reg_loss_r = metaregular((rd_drugEmbedding[uid.cpu().numpy()]), drugEmbedding, self.rrMat[uid.cpu().numpy()])
            reg_loss_d = metaregular((rd_disEmbedding[iid.cpu().numpy()]), disEmbedding, self.ddMat[iid.cpu().numpy()])
            meta_reg_loss = (reg_loss_r + reg_loss_d) / 2.0

        drugEmbeddingAll = self.args.wr1 * rd_drugEmbedding + self.args.wr2 * drugEmbedding
        disEmbeddingAll = self.args.wd1 * rd_disEmbedding + self.args.wd2 * disEmbedding

        return (drugEmbedding, disEmbedding, drugEmbeddingAll, disEmbeddingAll, rd_drugEmbedding, rd_disEmbedding,
                meta_reg_loss, all_rd_embeddings)
