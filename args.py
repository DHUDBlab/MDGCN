import argparse


def make_args():
    parser = argparse.ArgumentParser(description='DR main.py')
    parser.add_argument('--dataset', type=str, default='Fdataset')
    parser.add_argument('--batch', type=int, default=4096, metavar='N', help='input batch size for training')
    parser.add_argument('--seed', type=int, default=123, metavar='int', help='random seed')
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train')
    parser.add_argument('--hide_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--min_lr', type=float, default=0.0001)

    parser.add_argument('--decay', type=float, default=0.97, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type=float, default=0.067, metavar='LR', help='learning rate')
    parser.add_argument('--layers', type=int, default=5, help='the numbers of GCN layer')
    parser.add_argument('--rank', type=int, default=4, help='the dimension of low rank matrix decomposition')
    parser.add_argument('--topK', type=int, default=5, help='num_neighbor')

    parser.add_argument('--ssl_beta', type=float, default=0.1, help='weight of loss with ssl')
    parser.add_argument('--ssl_reg_r', type=float, default=0.068)  # drug reg
    parser.add_argument('--ssl_reg_d', type=float, default=0.088)  # disease reg

    parser.add_argument('--wr1', type=float, default=1.0, help='the coefficient of feature fusion ')
    parser.add_argument('--wr2', type=float, default=0.0, help='the coefficient of feature fusion')
    parser.add_argument('--wd1', type=float, default=1.0, help='the coefficient of feature fusion ')
    parser.add_argument('--wd2', type=float, default=0.0, help='the coefficient of feature fusion')

    parser.add_argument('--metareg', type=float, default=0.15, help='weight of loss with reg')
    parser.add_argument('--ssl_temp', type=float, default=0.5, help='the temperature in softmax')
    return parser.parse_args()


args = make_args()
