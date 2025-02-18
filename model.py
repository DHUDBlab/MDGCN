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
        # Squeeze: 对节点特征进行全局池化（取特征均值）
        z = torch.mean(x, dim=0)  # 全局池化，输出为 [F]
        # Excitation: 两层全连接
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))  # 输出通道权重
        # Reweight: 权重加权
        return x * z


class MODEL(nn.Module):
    def __init__(self, args, drug_num, dis_num, rrMat, ddMat, rdMat, hide_dim, layer_num):
        super(MODEL, self).__init__()
        self.args = args
        self.drug_num = drug_num
        self.dis_num = dis_num
        self.rrMat = rrMat
        self.ddMat = ddMat
        self.rdMat = rdMat  # trainmat拼接大矩阵
        self.hide_dim = hide_dim  # embedding size = 64
        self.layer_num = layer_num

        # rd_mat = self.rdMat[: self.drug_num, self.drug_num:]  # (763,681),又变回了trainmat，为何？
        # values = t.FloatTensor(rd_mat.tocoo().data)
        # indices = np.vstack((rd_mat.tocoo().row, rd_mat.tocoo().col))
        # i = t.LongTensor(indices)
        # v = t.FloatTensor(values)
        # shape = rd_mat.tocoo().shape
        # _rd_mat = t.sparse.FloatTensor(i, v, t.Size(shape))
        # self.rd_adj = _rd_mat  # 稀疏张量
        # self.dr_adj = _rd_mat.transpose(0, 1)

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

    # def metafortransform(self, auxi_embed_r, target_embed_r, auxi_embed_d, target_embed_d):
    #
    #     # Neighbor information of the target node
    #     drug_neighbor = t.matmul(self.rd_adj.cuda(), target_embed_d)  # 初始的药-病张量 X 卷积后的药-病张量 = (763x64)
    #     dis_neighbor = t.matmul(self.dr_adj.cuda(), target_embed_r)  # 初始的药-病矩阵张量 X 卷积后的药-病矩阵张量 = (763x64)
    #
    #     # Meta-knowledge extraction
    #     tembed_r = (self.meta_net_r(
    #         t.cat((auxi_embed_r, target_embed_r, drug_neighbor), dim=1).detach()))  # 全连接层，（763*3，64）-> (763x64)
    #     tembed_d = (self.meta_net_d(t.cat((auxi_embed_d, target_embed_d, dis_neighbor), dim=1).detach()))
    #
    #     """ Personalized transformation parameter matrix """
    #     # Low rank matrix decomposition
    #     meta_r1 = self.mlp_r1(tembed_r).reshape(-1, self.hide_dim, self.k)  # d*k
    #     meta_r2 = self.mlp_r2(tembed_r).reshape(-1, self.k, self.hide_dim)  # k*d
    #     meta_d1 = self.mlp_d1(tembed_d).reshape(-1, self.hide_dim, self.k)  # d*k
    #     meta_d2 = self.mlp_d2(tembed_d).reshape(-1, self.k, self.hide_dim)  # k*d
    #     meta_bias_r1 = (t.mean(meta_r1, dim=0))
    #     meta_bias_r2 = (t.mean(meta_r2, dim=0))
    #     meta_bias_d1 = (t.mean(meta_d1, dim=0))
    #     meta_bias_d2 = (t.mean(meta_d2, dim=0))
    #     low_weight_r1 = F.softmax(meta_r1 + meta_bias_r1, dim=1)
    #     low_weight_r2 = F.softmax(meta_r2 + meta_bias_r2, dim=1)
    #     low_weight_d1 = F.softmax(meta_d1 + meta_bias_d1, dim=1)
    #     low_weight_d2 = F.softmax(meta_d2 + meta_bias_d2, dim=1)
    #
    #     # The learned matrix as the weights of the transformed network
    #     tembed_rs = (t.sum(t.multiply(auxi_embed_r.unsqueeze(-1), low_weight_r1),
    #                        dim=1))  # Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
    #     tembed_rs = t.sum(t.multiply(tembed_rs.unsqueeze(-1), low_weight_r2), dim=1)
    #     tembed_ds = (t.sum(t.multiply(auxi_embed_d.unsqueeze(-1), low_weight_d1), dim=1))
    #     tembed_ds = t.sum(t.multiply(tembed_ds.unsqueeze(-1), low_weight_d2), dim=1)
    #     trans_drugEmbed = tembed_rs
    #     trans_disEmbed = tembed_ds
    #     return trans_drugEmbed, trans_disEmbed

    def forward(self, ifTraining, uid, iid, norm=1):

        # 1，制作最初的3个嵌入
        dis_index = np.arange(0, self.dis_num)
        drug_index = np.arange(0, self.drug_num)
        rd_index = np.array(drug_index.tolist() + [i + self.drug_num for i in dis_index])  # 1444

        # Initialize Embeddings
        drug_embed0 = self.embedding_dict['drug_emb'].weight
        dis_embed0 = self.embedding_dict['dis_emb'].weight
        # drug Emb, dis Emb, drug-dis Emb
        rr_embed0 = self.self_gating_r(drug_embed0)
        dd_embed0 = self.self_gating_d(dis_embed0)
        rd_embeddings = t.cat([drug_embed0, dis_embed0], 0)  # 只表示了[药-药，病-病]

        all_drug_embeddings = [rr_embed0]
        all_dis_embeddings = [dd_embed0]
        all_rd_embeddings = [rd_embeddings]

        # 2.执行GCN
        # Encoder
        for i in range(len(self.encoder)):
            # 2.1执行一次图卷积
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

            # 执行SE模块(对整体准确率稍稍提升，对提升aupr有用，但增加复杂度，用的时候直接取消注释就好，只用第一个，其他两个差)
            #  rdEmbeddings0 = self.se_rd(rdEmbeddings0)  # 1
            # rdEmbeddings0 = self.se_rd(drugEmbeddings)  # 2
            # rdEmbeddings0 = self.se_rd(disEmbeddings)  # 3

            rd_random_noise = t.rand_like(rdEmbeddings0).cuda()
            rdEmbeddings0 += t.sign(rdEmbeddings0) * F.normalize(rd_random_noise, dim=-1) * self.args.eps
            # 2.2每一层GCN中都将卷积后的大矩阵分割成2部分，分别加到药物和疾病嵌入上
            rd_drugEmbedding0, rd_disEmbedding0 = t.split(rdEmbeddings0, [self.drug_num, self.dis_num])

            # 调这个权重没啥用，但有影响
            drugEd = (0.5 * drugEmbeddings0 + 0.5 * rd_drugEmbedding0)
            disEd = (0.5 * disEmbeddings0 + 0.5 * rd_disEmbedding0)

            # 记录，传到下一层
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

        # 对所有嵌入进行加权平均（注释掉的都是加权平均的东西，没啥用，还是用原来的平均比较好）
        # weights = t.FloatTensor([0.1, 0.08, 0.2, 0.2, 0.1, 0.08, 0.08, 0.08, 0.08])
        # device = drugEmbedding.device  # 获取 drugEmbedding 的设备
        # weights = weights.to(device)
        # 对所有嵌入进行加权平均
        # drugEmbedding = t.sum(drugEmbedding * weights.view(1, -1, 1), dim=1)
        # disEmbedding = t.sum(disEmbedding * weights.view(1, -1, 1), dim=1)
        # rdEmbedding = t.sum(rdEmbedding * weights.view(1, -1, 1), dim=1)

        #  现在这12（比层数多1），和9（比层数多1）应该是最好的了，不用调
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
