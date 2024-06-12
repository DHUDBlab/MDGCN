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

from data import DataHandler
from model import MODEL
# from args import make_args
from utils import log, MyTransData, ssl_loss, Colors

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")

isLoadModel = False

def make_args():
    parser = argparse.ArgumentParser(description='DR main.py')
    parser.add_argument('--dataset', type=str, default='lrssl')
    parser.add_argument('--batch', type=int, default=8192, metavar='N', help='input batch size for training')
    parser.add_argument('--seed', type=int, default=123, metavar='int', help='random seed')
    parser.add_argument('--epochs', type=int, default=0, metavar='N', help='number of epochs to train')
    parser.add_argument('--hide_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--min_lr', type=float, default=0.0001)

    parser.add_argument('--decay', type=float, default=0.99, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type=float, default=0.055, metavar='LR', help='learning rate')
    parser.add_argument('--layers', type=int, default=9, help='the numbers of GCN layer')
    parser.add_argument('--rank', type=int, default=6, help='the dimension of low rank matrix decomposition')
    parser.add_argument('--topK', type=int, default=6, help='num_neighbor')

    parser.add_argument('--ssl_beta', type=float, default=0.1, help='weight of loss with ssl')
    parser.add_argument('--ssl_reg_r', type=float, default=0.08)  # drug reg
    parser.add_argument('--ssl_reg_d', type=float, default=0.09)  # disease reg

    parser.add_argument('--wr1', type=float, default=0.8, help='the coefficient of feature fusion ')
    parser.add_argument('--wr2', type=float, default=0.2, help='the coefficient of feature fusion')
    parser.add_argument('--wd1', type=float, default=0.8, help='the coefficient of feature fusion ')
    parser.add_argument('--wd2', type=float, default=0.2, help='the coefficient of feature fusion')

    parser.add_argument('--metareg', type=float, default=0.15, help='weight of loss with reg')
    parser.add_argument('--ssl_temp', type=float, default=0.5, help='the temperature in softmax')
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
            ssl_loss_drug = ssl_loss(rd_drugEmbed, drugEmbed, drugs, self.args.ssl_temp)
            ssl_loss_dis = ssl_loss(rd_disEmbed, disEmbed, diseases, self.args.ssl_temp)
            ssl_loss_all = args.ssl_reg_r * ssl_loss_drug \
                    + args.ssl_reg_d * ssl_loss_dis

            # prediction
            preds = self.predict_model(rd_drugEmbedAll[drugs], rd_disEmbedAll[diseases])
            bce_loss = self.bce_loss(preds, labels)
            epLoss += bce_loss.item()
            # regLoss = (t.norm(rd_drugEmbedAll[drugs]) ** 2 + t.norm(rd_disEmbedAll[diseases]) ** 2)

            regLoss = (1 / 2) * (t.norm(rd_drugEmbedAll[drugs], p=2).pow(2) + t.norm(rd_disEmbedAll[diseases], p=2).pow(2))
            regLoss = (t.norm(rd_drugEmbedAll[drugs], p=2).pow(2) + t.norm(rd_disEmbedAll[diseases], p=2).pow(2)) / float(len(drugs))
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
        self.cur_Epoch = 0
        best_auroc = -1
        best_aupr = -1
        best_epoch = -1
        AUROC_lis = []
        for e in range(args.epochs + 1):
            self.cur_Epoch = e
            loss = self._train()
            AUROC, AUPR = self._test()
            log("epoch %d/%d, loss=%.4f, AUROC=%.4f, AUPR=%.4f" % (e, args.epochs, loss, AUROC, AUPR))
            # log("epoch %d/%d, loss=%.4f, AUROC=%.3f, AUPR=%.3f" % (e, args.epochs, loss, AUROC, AUPR))
            self.train_loss.append(loss)
            self.test_auroc.append(AUROC)
            self.test_aupr.append(AUPR)
            self.adjust_learning_rate()
            cur_AUROC = round(AUROC, 4)
            cur_AUPR = round(AUPR, 4)
            if cur_AUROC > round(best_auroc, 4) or cur_AUROC == round(best_auroc, 4) and cur_AUPR > round(best_aupr, 4):
                best_epoch, best_auroc, best_aupr = e, cur_AUROC, cur_AUPR
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
        log("best epoch %d, AUROC=%.4f, AUPR=%.4f" % (best_epoch, best_auroc, best_aupr), color=Colors.MAGENTA)
        # log("best epoch = %d, AUROC= %.3f, AUPR=%.3f" % (best_epoch, best_auroc, best_aupr))
        # log("------------------------------------------------------")
        return best_epoch, best_auroc, best_aupr


if __name__ == '__main__':
    args = make_args()
    dataset = args.dataset
    dataHandler = DataHandler(args)
    results_dict = {'ep': [], 'auroc': [], 'aupr': []}  # 3*10*10
    for time in range(1, 2):
        print(f"{Colors.RED}++++++++++++++++++++++ {str(time)} time START +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++{Colors.RESET}")
        time_results = {'ep': [], 'auroc': [], 'aupr': []}
        for fold in range(1, 2):
            print(f"{Colors.YELLOW}======================== {str(fold)} fold start ============================================={Colors.RESET}")
            data = dataHandler.data[0][fold-1]
            # data = dataHandler.data[0][9]
            distanceMat = [dataHandler.data[1], dataHandler.data[2], data[-1]]
            hope = Hope(args, data, distanceMat)
            ep, auroc, aupr = hope.run()
            # record result
            time_results['ep'].append(ep)  # 10
            time_results['auroc'].append(auroc)
            time_results['aupr'].append(aupr)
            print(f"{Colors.YELLOW}======================== {str(fold)} fold end ============================================={Colors.RESET}\n")
        # time_results_df = pd.DataFrame(time_results)
        # time_results_df.to_csv(os.path.join('result', args.dataset + f'_{time}times_10folds.csv'), index=False)
        print(f"{Colors.YELLOW}{time} time auroc list: {time_results['auroc']}{Colors.RESET}")
        print(f"{Colors.YELLOW}{time} time aupr list: {time_results['aupr']}{Colors.RESET}")
        log(f"mean auc: {np.mean(time_results['auroc']):.4f}±{np.std(time_results['auroc']):.4f}, mean aupr: {np.mean(time_results['aupr']):.4f}±{np.std(time_results['aupr']):.4f}",color=Colors.YELLOW)
        log(f"mean auc: {np.mean(time_results['auroc']):.3f}±{np.std(time_results['auroc']):.3f}, mean aupr: {np.mean(time_results['aupr']):.3f}±{np.std(time_results['aupr']):.3f}",color=Colors.YELLOW)
        
        results_dict['ep'].extend(time_results['ep'])  # 10*10
        results_dict['auroc'].extend(time_results['auroc'])
        results_dict['aupr'].extend(time_results['aupr'])
        print(f"{Colors.RED}++++++++++++++++++++++ {str(time)} time END +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++{Colors.RESET}\n\n")
    
    # results_df = pd.DataFrame(results_dict)
    # results_df.to_csv(os.path.join('result', args.dataset + '_all_times_all_folds.csv'), index=False)

    print("10 times 10 cross validation testing finished !\n")
    log(f"Mean AUC: {np.mean(results_dict['auroc']):.4f}±{np.std(results_dict['auroc']):.4f}, Mean AUPR: {np.mean(results_dict['aupr']):.4f}±{np.std(results_dict['aupr']):.4f}", color=Colors.CYAN)
    log(f"Mean AUC: {np.mean(results_dict['auroc']):.3f}±{np.std(results_dict['auroc']):.3f}, Mean AUPR: {np.mean(results_dict['aupr']):.3f}±{np.std(results_dict['aupr']):.3f}", color=Colors.GREEN)
