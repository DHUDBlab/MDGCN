import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from layers import MLP, GCN_layer
from transformer_model import TransformerModel
from utils import metaregular, re_features


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
        self.transformer = TransformerModel(hops=args.hops,
                                            n_class=2,
                                            input_dim=hide_dim,
                                            pe_dim=args.pe_dim,
                                            n_layers=args.n_layers,
                                            num_heads=args.n_heads,
                                            hidden_dim=args.trans_hidden_dim,
                                            ffn_dim=args.ffn_dim,
                                            dropout_rate=args.dropout,
                                            attention_dropout_rate=args.attention_dropout).cuda()

        rd_mat = self.rdMat[: self.drug_num, self.drug_num:]  # [269, 598]
        values = t.FloatTensor(rd_mat.tocoo().data)
        indices = np.vstack((rd_mat.tocoo().row, rd_mat.tocoo().col))
        i = t.LongTensor(indices)
        v = t.FloatTensor(values)
        shape = rd_mat.tocoo().shape
        _rd_mat = t.sparse.FloatTensor(i, v, t.Size(shape))
        self.rd_adj = _rd_mat  # [269, 598]
        self.dr_adj = _rd_mat.transpose(0, 1)  # [598, 269]

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

    def self_gating_r(self, em):
        return t.multiply(em, t.sigmoid(t.matmul(em, self.gating_weight_r) + self.gating_weight_rb))

    def self_gating_d(self, em):
        return t.multiply(em, t.sigmoid(t.matmul(em, self.gating_weight_d) + self.gating_weight_db))

    def metafortransform(self, auxi_embed_r, target_embed_r, auxi_embed_d, target_embed_d):

        # Neighbor information of the target node
        drug_neighbor = t.matmul(self.rd_adj.cuda(), target_embed_d)
        dis_neighbor = t.matmul(self.dr_adj.cuda(), target_embed_r)

        # Meta-knowledge extraction
        tembed_r = (self.meta_net_r(t.cat((auxi_embed_r, target_embed_r, drug_neighbor), dim=1).detach()))
        tembed_d = (self.meta_net_d(t.cat((auxi_embed_d, target_embed_d, dis_neighbor), dim=1).detach()))

        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        meta_r1 = self.mlp_r1(tembed_r).reshape(-1, self.hide_dim, self.k)  # d*k
        meta_r2 = self.mlp_r2(tembed_r).reshape(-1, self.k, self.hide_dim)  # k*d
        meta_d1 = self.mlp_d1(tembed_d).reshape(-1, self.hide_dim, self.k)  # d*k
        meta_d2 = self.mlp_d2(tembed_d).reshape(-1, self.k, self.hide_dim)  # k*d
        meta_bias_r1 = (t.mean(meta_r1, dim=0))
        meta_bias_r2 = (t.mean(meta_r2, dim=0))
        meta_bias_d1 = (t.mean(meta_d1, dim=0))
        meta_bias_d2 = (t.mean(meta_d2, dim=0))
        low_weight_r1 = F.softmax(meta_r1 + meta_bias_r1, dim=1)
        low_weight_r2 = F.softmax(meta_r2 + meta_bias_r2, dim=1)
        low_weight_d1 = F.softmax(meta_d1 + meta_bias_d1, dim=1)
        low_weight_d2 = F.softmax(meta_d2 + meta_bias_d2, dim=1)

        # The learned matrix as the weights of the transformed network
        tembed_rs = (t.sum(t.multiply(auxi_embed_r.unsqueeze(-1), low_weight_r1),
                           dim=1))  # Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
        tembed_rs = t.sum(t.multiply(tembed_rs.unsqueeze(-1), low_weight_r2), dim=1)
        tembed_ds = (t.sum(t.multiply(auxi_embed_d.unsqueeze(-1), low_weight_d1), dim=1))
        tembed_ds = t.sum(t.multiply(tembed_ds.unsqueeze(-1), low_weight_d2), dim=1)
        trans_drugEmbed = tembed_rs
        trans_disEmbed = tembed_ds
        return trans_drugEmbed, trans_disEmbed

    def forward(self, ifTraining, uid, iid, norm=1):
        dis_index = np.arange(0, self.dis_num)
        drug_index = np.arange(0, self.drug_num)
        rd_index = np.array(drug_index.tolist() + [i + self.drug_num for i in dis_index])

        # Initialize Embeddings
        drug_embed0 = self.embedding_dict['drug_emb'].weight  # [269， 64]
        dis_embed0 = self.embedding_dict['dis_emb'].weight  # [598， 64]
        # drug Emb, dis Emb, drug-dis Emb
        rr_embed0 = self.self_gating_r(drug_embed0)
        dd_embed0 = self.self_gating_d(dis_embed0)
        rd_embeddings = t.cat([drug_embed0, dis_embed0], 0)  # [867, 64]
        # Record every layer's output including initial emb
        all_drug_embeddings = [rr_embed0]
        all_dis_embeddings = [dd_embed0]
        all_rd_embeddings = [rd_embeddings]

        # Transformer Input
        nodes_features = re_features(self.rdMat, rd_embeddings, self.args.hops).cuda()  # [867, 867]  [867, 64] 1
        tfr_rdEmbeddings = self.transformer(nodes_features)
        tfr_rd_drugEmbedding, tfr_rd_disEmbedding = t.split(tfr_rdEmbeddings, [self.drug_num, self.dis_num])
        """
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

            # Aggregate message features across the two related views in the middle layer then fed into the next layer
            rd_drugEmbedding0, rd_disEmbedding0 = t.split(rdEmbeddings0, [self.drug_num, self.dis_num])
            drugEd = (drugEmbeddings0 + rd_drugEmbedding0) / 2.0
            disEd = (disEmbeddings0 + rd_disEmbedding0) / 2.0
            # next layer input
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
                all_dis_embeddings += [norm_embeddings]
                all_rd_embeddings += [norm_embeddings]
        drugEmbedding = t.stack(all_drug_embeddings, dim=1)
        drugEmbedding = t.mean(drugEmbedding, dim=1)
        disEmbedding = t.stack(all_dis_embeddings, dim=1)
        disEmbedding = t.mean(disEmbedding, dim=1)
        rdEmbedding = t.stack(all_rd_embeddings, dim=1)
        rdEmbedding = t.mean(rdEmbedding, dim=1)
        rd_drugEmbedding, rd_disEmbedding = t.split(rdEmbedding, [self.drug_num, self.dis_num])

        # Cross-View Fusion Features
        metats_drug_embed, metats_dis_embed = self.metafortransform(drugEmbedding, rd_drugEmbedding, disEmbedding,
                                                                    rd_disEmbedding)
        drugEmbedding = drugEmbedding + metats_drug_embed
        disEmbedding = disEmbedding + metats_dis_embed

        # Regularization: the constraint of transformed reasonableness
        meta_reg_loss = 0
        if ifTraining:
            reg_loss_r = metaregular((rd_drugEmbedding[uid.cpu().numpy()]), drugEmbedding,
                                     self.rrMat[uid.cpu().numpy()])
            reg_loss_d = metaregular((rd_disEmbedding[iid.cpu().numpy()]), disEmbedding, self.ddMat[iid.cpu().numpy()])
            meta_reg_loss = (reg_loss_r + reg_loss_d) / 2.0
        return drugEmbedding, disEmbedding, (self.args.wr1 * rd_drugEmbedding + self.args.wr2 * drugEmbedding), (
                    self.args.wd1 * rd_disEmbedding + self.args.wd2 * disEmbedding), rd_drugEmbedding, rd_disEmbedding, meta_reg_loss
        """
        return tfr_rd_drugEmbedding, tfr_rd_disEmbedding