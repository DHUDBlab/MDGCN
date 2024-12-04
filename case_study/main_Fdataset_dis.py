import argparse

import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import numpy as np
import time
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.utils.data as dataloader
import torch.nn.functional as F

from case_data import CaseDataHandler
from model import MODEL
# from args import make_args
from utils import log, MyTransData, ssl_loss, Colors, metaregular, common_loss, case_result

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")

isLoadModel = False

def make_args():
    parser = argparse.ArgumentParser(description='DR main.py')
    parser.add_argument('--dataset', type=str, default='Fdataset')
    parser.add_argument('--batch', type=int, default=4096, metavar='N', help='input batch size for training')
    parser.add_argument('--seed', type=int, default=123, metavar='int', help='random seed')
    parser.add_argument('--epochs', type=int, default=180, metavar='N', help='number of epochs to train')
    parser.add_argument('--hide_dim', type=int, default=128, metavar='N', help='embedding size')
    parser.add_argument('--min_lr', type=float, default=0.0001)

    parser.add_argument('--decay', type=float, default=0.99, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate')
    parser.add_argument('--layers', type=int, default=5+3, help='the numbers of GCN layer')
    parser.add_argument('--rank', type=int, default=4, help='the dimension of low rank matrix decomposition')
    parser.add_argument('--topK', type=int, default=3, help='num_neighbor')

    parser.add_argument('--ssl_beta', type=float, default=0.1, help='weight of loss with ssl')
    parser.add_argument('--ssl_reg_r', type=float, default=0.068)  # drug reg
    parser.add_argument('--ssl_reg_d', type=float, default=0.088)  # disease reg

    parser.add_argument('--wr1', type=float, default=0.9, help='the coefficient of feature fusion ')
    parser.add_argument('--wr2', type=float, default=0.1, help='the coefficient of feature fusion')
    parser.add_argument('--wd1', type=float, default=0.9, help='the coefficient of feature fusion ')
    parser.add_argument('--wd2', type=float, default=0.1, help='the coefficient of feature fusion')

    parser.add_argument('--metareg', type=float, default=0.19, help='weight of loss with reg')
    parser.add_argument('--ssl_temp', type=float, default=0.5, help='the temperature in softmax')
    parser.add_argument('--new1', type=float, default=0.9, help='parser_1')
    parser.add_argument('--new2', type=float, default=0.01, help='parser_2')
    parser.add_argument('--eps', type=float, default=0.3, help='noise')

    # parser.add_argument('--com_beta', type=float, default=0.1, help='weight of loss with common loss')
    # parser.add_argument('--com_reg_r', type=float, default=0.1)  # drug reg
    # parser.add_argument('--com_reg_d', type=float, default=0.1)  # disease reg

    parser.add_argument('--all_case_name', type=list, default=['ParkinsonDisease', 'AlzheimerDisease'], help='case study') # ['ParkinsonDisease', 'BreastCancer', 'AlzheimerDisease']
    parser.add_argument('--all_case_id', type=list, default=[119, 7], help='case study') # [119, 19, 7]
    # parser.add_argument('--all_case_name', type=list, default=['ParkinsonDisease'], help='case study')
    # parser.add_argument('--all_case_id', type=list, default=[119], help='case study')
    return parser.parse_args()

class Hope:
    def __init__(self, args, _data, _distanceMat):
        self.args = args
        self.rrMat, self.ddMat, self.rdMat = _distanceMat
        self.trainMat, self.testMat, _ = _data
        self.drugNum, self.disNum = self.trainMat.shape
        self.train_loader = dataloader.DataLoader(MyTransData(self.trainMat), batch_size=args.batch, shuffle=True, num_workers=0)
        self.train_loss = []
        self.test_auroc = []
        self.test_aupr = []

    def prepare_model(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(args.seed)
        self.model = MODEL(
            self.args,
            self.drugNum,
            self.disNum,
            self.rrMat, self.ddMat, self.rdMat,
            args.hide_dim,
            args.layers,
        ).cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=args.lr)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def predict_model(self, r, d, isTest=False):
        pred = t.sum(r * d, dim=1)
        return pred

    def adjust_learning_rate(self):
        if self.opt != None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * args.decay, args.min_lr)

    def get_model_params(self):
        # title = "DrugRep" + "_"
        ModelParams = args.dataset + ", " + \
                      "layers " + str(args.layers) + "," + \
                      "topK " + str(args.topK) + "," + \
                      "new1 " + str(args.new1) + "," + \
                      "eps " + str(args.eps) + "," + \
                      "w1 " + str(args.wr1) + "," + \
                      "w2 " + str(args.wr2) + "," + \
                      "new2 " + str(args.new2) + "," + \
                      "sslr " + str(args.ssl_reg_r) + "," + \
                      "ssld " + str(args.ssl_reg_d) + "," +\
                      "decay " + str(args.decay) + "," + \
                      "lr " + str(args.lr)
        return ModelParams

    def save_history(self):
        history = dict()
        history['loss'] = self.train_loss
        history['auroc'] = self.test_auroc
        history['aupr'] = self.test_aupr
        ModelName = self.get_model_params()

        # make history dir (exist_ok=True will not make dir)
        history_folder = './History/' + dataset + '/'
        os.makedirs(history_folder, exist_ok=True)
        with open(history_folder + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def save_model(self):
        ModelName = self.get_model_params()
        history = dict()
        history['loss'] = self.train_loss
        history['auroc'] = self.test_auroc
        history['aupr'] = self.test_aupr
        savePath = r'./Model/' + dataset + r'/' + ModelName + r'.pth'
        params = {
            'model': self.model,
            'epoch': self.cur_Epoch,
            'args': args,
            'opt': self.opt,
            'history': history
        }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def load_model(self, modelPath):
        checkpoint = t.load(r'./Model/' + dataset + r'/' + modelPath + r'.pth')
        self.cur_Epoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.test_auroc = history['auroc']
        self.test_aupr = history['aupr']
        log("load model %s in epoch %d" % (modelPath, checkpoint['epoch']))

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = t.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def ssl1_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb_all, context_item_emb_all = t.split(context_emb, [self.drugNum, self.disNum])
        initial_user_emb_all, initial_item_emb_all = t.split(initial_emb, [self.drugNum, self.disNum])
        user_cl_loss = self.InfoNCE(context_user_emb_all[user], initial_user_emb_all[user], self.args.ssl_temp)
        item_cl_loss = self.InfoNCE(context_item_emb_all[item], initial_item_emb_all[item], self.args.ssl_temp)
        return user_cl_loss + item_cl_loss

    def _train(self):
        epLoss = 0
        steps = self.train_loader.dataset.__len__() // args.batch
        for i, tem in enumerate(self.train_loader):
            drugs, diseases, labels = tem  # (x,y) value

            drugs = drugs.long().cuda()
            diseases = diseases.long().cuda()
            labels = labels.cuda().float()
            self.ifTraining = True
            # GET Embedding
            drugEmbed, disEmbed, rd_drugEmbedAll, rd_disEmbedAll, rd_drugEmbed, rd_disEmbed, meta_reg_loss, all_rd_embeddings = self.model(
                self.ifTraining, drugs, diseases, norm=1)  # 这里输出比以前多后面三个

            # Contrastive Learning of collaborative relations
            ############### 新加的
            # 自己和自己对比
            initial_emb1 = all_rd_embeddings[0]
            context_emb1 = all_rd_embeddings[self.args.layers]
            ssl1_loss = self.ssl1_layer_loss(context_emb1, initial_emb1, drugs, diseases)

            # initial_emb2 = all_rd_embeddings[0]
            # context_emb2 = all_rd_embeddings[4]
            # ssl2_loss = self.ssl1_layer_loss(context_emb2, initial_emb2, drugs, diseases)
            ###############

            ssl_loss_drug = ssl_loss(rd_drugEmbed, drugEmbed, drugs, self.args.ssl_temp)
            ssl_loss_dis = ssl_loss(rd_disEmbed, disEmbed, diseases, self.args.ssl_temp)
            ssl_loss_all = self.args.new1 * (args.ssl_reg_r * ssl_loss_drug + args.ssl_reg_d * ssl_loss_dis + self.args.new2 * ssl1_loss)
            # 参数1：0.6或0.7最好，对正确率和AUPR影响很大，越小收敛越慢（多调这个）； 参数2：0.01最好（这两个参数都是分开调的，组合起来不一定是最好）


            # common_loss_drug = common_loss(rd_drugEmbed[drugs], drugEmbed[drugs])
            # common_loss_dis = common_loss(rd_disEmbed[diseases], disEmbed[diseases])
            # common_loss_all = args.com_reg_r * common_loss_drug + args.com_reg_d * common_loss_dis
            # prediction
            preds = self.predict_model(rd_drugEmbedAll[drugs], rd_disEmbedAll[diseases])
            bce_loss = self.bce_loss(preds, labels)
            epLoss += bce_loss.item()

            loss = bce_loss + ssl_loss_all * args.ssl_beta + meta_reg_loss * args.metareg #+ common_loss_all * args.com_beta
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
        return epLoss / steps

    def _test(self):
        with t.no_grad():
            rid = np.arange(0, self.drugNum)
            did = np.arange(0, self.disNum)
            pairs, labels = tuple([self.testMat.row, self.testMat.col]), self.testMat.data.tolist()
            test_r, test_d = pairs
            self.ifTraining = False
            _, _, rd_drugEmbed, rd_disEmbed, _, _, _, _ = self.model(self.ifTraining, rid, did, norm=1)
            preds = t.sigmoid(self.predict_model(rd_drugEmbed[test_r], rd_disEmbed[test_d], isTest=True)).detach().cpu().numpy().tolist()
            epAuc, epAupr = roc_auc_score(labels, preds), average_precision_score(labels, preds)
            return epAuc, epAupr, labels, preds

    def run(self):
        self.prepare_model()
        self.cur_Epoch = 0
        best_auroc = -1
        best_aupr = -1
        best_epoch = -1
        true, score = [], []
        for e in range(args.epochs + 1):
            self.cur_Epoch = e
            loss = self._train()
            AUROC, AUPR, y_true, y_score = self._test()
            log("epoch %d/%d, loss=%.4f, AUROC=%.4f, AUPR=%.4f" % (e, args.epochs, loss, AUROC, AUPR))
            # log("epoch %d/%d, loss=%.4f, AUROC=%.3f, AUPR=%.3f" % (e, args.epochs, loss, AUROC, AUPR))
            self.train_loss.append(loss)
            self.test_auroc.append(AUROC)
            self.test_aupr.append(AUPR)
            self.adjust_learning_rate()
            cur_AUROC = round(AUROC, 4)
            cur_AUPR = round(AUPR, 4)
            if cur_AUROC > round(best_auroc, 4) or cur_AUROC == round(best_auroc, 4) and cur_AUPR > round(best_aupr, 4):
                best_epoch, best_auroc, best_aupr, true, score = e, cur_AUROC, cur_AUPR, y_true, y_score
                # self.save_model()
            """
            self.save_history()
            if wait==self.args.patience:
                log('Early stop! best epoch = %d'%(best_epoch))
                # self.load_model(self.get_model_params())
                break
            """
        # log("------------------------------------------------------")
        print('\nparams: ' + self.get_model_params())
        log("best epoch %d, AUROC=%.4f, AUPR=%.4f" % (best_epoch, best_auroc, best_aupr), color=Colors.MAGENTA)
        # log("best epoch = %d, AUROC= %.3f, AUPR=%.3f" % (best_epoch, best_auroc, best_aupr))
        # log("------------------------------------------------------")
        _result = {
            "y_true": true,
            "y_score": score
        }
        return best_epoch, best_auroc, best_aupr, _result


if __name__ == '__main__':
    args = make_args()
    dataset = args.dataset
    dataHandler = CaseDataHandler(args)
    print(f"{Colors.RED}++++++++++++++++++++++ START +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++{Colors.RESET}")
    for index, case in enumerate(args.all_case_name):
        print(f"{Colors.YELLOW}======================== {str(case)} start ============================================={Colors.RESET}")
        data = dataHandler.data[0][index]
        distanceMat = [dataHandler.data[1], dataHandler.data[2], data[-1]]  # rrMat, ddMat, heteroMat
        hope = Hope(args, data, distanceMat)
        ep, auroc, aupr, result = hope.run()
        case_result(case, dataHandler.drug_id, result["y_score"])
        print(f"{Colors.YELLOW}======================== {str(case)} end ============================================={Colors.RESET}\n")
    print(f"{Colors.RED}++++++++++++++++++++++ END +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++{Colors.RESET}\n\n")
