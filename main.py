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

from data import DataHandler
from model import MODEL
from args import args
from utils import log, MyTransData, ssl_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")

isLoadModel = False


class Hope:
    def __init__(self, data, distanceMat):
        self.rrMat, self.ddMat, self.rdMat = distanceMat
        self.trainMat, self.testMat, _ = data
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
        self.model = MODEL(
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
        ModelParams = args.dataset + " | " + \
                    "decay_" + str(args.decay) + " | " + \
                    "lr_" + str(args.lr) + " | " + \
                    "layers_" + str(args.layers) + " | " + \
                    "rank_" + str(args.rank) + " | " + \
                    "topK_" + str(args.topK) + " | " + \
                    "ssl_reg_r_" + str(args.ssl_reg_r) + " | " + \
                    "ssl_reg_d_" + str(args.ssl_reg_d)
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
            'epoch': self.curEpoch,
            'args': args,
            'opt': self.opt,
            'history': history
        }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def load_model(self, modelPath):
        checkpoint = t.load(r'./Model/' + dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.test_auroc = history['auroc']
        self.test_aupr = history['aupr']
        log("load model %s in epoch %d" % (modelPath, checkpoint['epoch']))

    def _train(self):
        epLoss = 0
        steps = self.train_loader.dataset.__len__() // args.batch
        for i, tem in enumerate(self.train_loader):
            drugs, diseases, labels = tem
            drugs = drugs.long().cuda()
            diseases = diseases.long().cuda()
            labels = labels.cuda().float()
            self.ifTraining = True
            # GET Embedding
            drugEmbed, disEmbed, rd_drugEmbedAll, rd_disEmbedAll, rd_drugEmbed, rd_disEmbed, meta_reg_loss = self.model(self.ifTraining, drugs, diseases, norm=1)
            # Contrastive Learning of collaborative relations
            ssl_loss_drug = ssl_loss(rd_drugEmbed, drugEmbed, drugs)
            ssl_loss_dis = ssl_loss(rd_disEmbed, disEmbed, diseases)
            ssl_loss_all = args.ssl_reg_r * ssl_loss_drug \
                    + args.ssl_reg_d * ssl_loss_dis

            # prediction
            preds = self.predict_model(rd_drugEmbedAll[drugs], rd_disEmbedAll[diseases])
            bce_loss = self.bce_loss(preds, labels)
            epLoss += bce_loss.item()
            # regLoss = (t.norm(rd_drugEmbedAll[drugs]) ** 2 + t.norm(rd_disEmbedAll[diseases]) ** 2)
            # loss = ((bce_loss + regLoss * args.reg ) / args.batch) + ssl_loss*args.ssl_beta + meta_reg_loss*args.metareg
            # loss = bce_loss # + regLoss * args.reg
            loss = bce_loss + ssl_loss_all * args.ssl_beta + meta_reg_loss * args.metareg  # + regLoss * args.reg
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
        return epLoss / steps

    def _test(self):
        with t.no_grad():
            rid = np.arange(0, self.drugNum)
            did = np.arange(0, self.disNum)
            pairs, labels = tuple([self.testMat.row, self.testMat.col]), self.testMat.data
            test_r, test_d = pairs
            self.ifTraining = False
            _, _, rd_drugEmbed, rd_disEmbed, _, _, _ = self.model(self.ifTraining, rid, did, norm=1)
            preds = self.predict_model(rd_drugEmbed[test_r], rd_disEmbed[test_d], isTest=True).detach().cpu()
            epAuc, epAupr = roc_auc_score(labels, preds), average_precision_score(labels, preds)
            return epAuc, epAupr

    def run(self):
        self.prepare_model()
        self.curEpoch = 0
        best_auroc = -1
        best_aupr = -1
        best_epoch = -1
        AUROC_lis = []
        for e in range(args.epochs + 1):
            self.curEpoch = e
            loss = self._train()
            AUROC, AUPR = self._test()
            log("epoch %d/%d, loss=%.4f, AUROC=%.4f, AUPR=%.4f" % (e, args.epochs, loss, AUROC, AUPR))
            # log("epoch %d/%d, loss=%.4f, AUROC=%.3f, AUPR=%.3f" % (e, args.epochs, loss, AUROC, AUPR))
            self.train_loss.append(loss)
            self.test_auroc.append(AUROC)
            self.test_aupr.append(AUPR)
            self.adjust_learning_rate()
            if AUROC >= best_auroc:
                best_epoch, best_auroc, best_aupr = e, AUROC, AUPR
                # self.save_model()
            AUROC_lis.append(AUROC)
            """
            self.save_history()
            if wait==self.args.patience:
                log('Early stop! best epoch = %d'%(best_epoch))
                # self.load_model(self.get_model_params())
                break
            """
        # log("------------------------------------------------------")
        print('params: ' + self.get_model_params())
        log("best epoch = %d, AUROC= %.4f, AUPR=%.4f" % (best_epoch, best_auroc, best_aupr))
        # log("best epoch = %d, AUROC= %.3f, AUPR=%.3f" % (best_epoch, best_auroc, best_aupr))
        # log("------------------------------------------------------")
        return best_epoch, best_auroc, best_aupr


if __name__ == '__main__':
    dataset = args.dataset
    dataHandler = DataHandler()
    # results_dict = {'ep': [], 'auroc': [], 'aupr': []}  # 3*10*10
    # max_results_dict = {'fold': [], 'ep': [], 'auroc': [], 'aupr': []}  # 4*10 best fold of every times
    for time in range(1, 11):
        print("********************** time", str(time), "**************************************************************************")
        time_results = {'ep': [], 'auroc': [], 'aupr': []}
        for fold in range(1, 11):
            print("======================== fold", str(fold), "=============================================")
            data = dataHandler.data[0][fold-1]
            # data = dataHandler.data[0][0]
            distanceMat = [dataHandler.data[1], dataHandler.data[2], data[-1]]
            hope = Hope(data, distanceMat)
            ep, auroc, aupr = hope.run()
            # record result
            time_results['ep'].append(ep)  # 10
            time_results['auroc'].append(auroc)
            time_results['aupr'].append(aupr)
        max_fold = np.argmax(time_results['auroc'])  # 1
        max_ep = time_results['ep'][max_fold]
        max_auroc = time_results['auroc'][max_fold]
        max_aupr = time_results['aupr'][max_fold]
        log("max fold = %d, AUROC= %.4f, AUPR=%.4f" % (max_fold, max_auroc, max_aupr))
        # results_dict['ep'].append(time_results['ep'])  # 10*10
        # results_dict['auroc'].append(time_results['auroc'])
        # results_dict['aupr'].append(time_results['aupr'])
        # max_results_dict['fold'].append(max_fold+1)  # 1*10
        # max_results_dict['ep'].append(max_ep)
        # max_results_dict['auroc'].append(max_auroc)
        # max_results_dict['aupr'].append(max_aupr)
    # results_df = pd.DataFrame(results_dict)
    # results_df.to_csv(os.path.join('result', args.dataset + '_10times_10folds.csv'), index=False)
    # max_results_df = pd.DataFrame(max_results_dict)
    # max_results_df.to_csv(os.path.join('result', args.dataset + '_10times_best_folds.csv'), index=False)
    #
    # print("********************************************************************************************************")
    # log(f"Mean AUC: {np.mean(max_results_dict['auroc']):.4f}±{np.std(max_results_dict['auroc']):.4f}, Mean AUPR: {np.mean(max_results_dict['aupr']):.4f}±{np.std(max_results_dict['aupr']):.4f}")
    # log(f"Mean AUC: {np.mean(max_results_dict['auroc']):.3f}±{np.std(max_results_dict['auroc']):.3f}, Mean AUPR: {np.mean(max_results_dict['aupr']):.3f}±{np.std(max_results_dict['aupr']):.3f}")
