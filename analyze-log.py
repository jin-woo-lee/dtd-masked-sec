import argparse
import os
from train import train
from test import test
import logging
import logging.handlers

def acc(list_str):
    list_float = []
    for st in list_str:
        acc = float(st.split(": ")[-1])
        list_float.append(acc)
    return list_float

if __name__=='__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str)
    args = parser.parse_args()
    f = open(args.log, 'r')
    logs = f.read().split('\n')
    top_1 = logs[1::3]
    top_5 = logs[2::3]

    top_1_0 = acc(top_1[0::3])    # no DTD
    top_1_1 = acc(top_1[1::3])    # 1st DTD
    top_1_2 = acc(top_1[2::3])    # 2nd DTD

    top_5_0 = acc(top_5[0::3])    # no DTD
    top_5_1 = acc(top_5[1::3])    # 1st DTD
    top_5_2 = acc(top_5[2::3])    # 2nd DTD
    print("top 1 acc")
    print("no DTD ", sum(top_1_0) / len(top_1_0))
    print("1st DTD", sum(top_1_1) / len(top_1_1))
    print("2nd DTD", sum(top_1_2) / len(top_1_2))
    print("top 5 acc")
    print("no DTD ", sum(top_5_0) / len(top_5_0))
    print("1st DTD", sum(top_5_1) / len(top_5_1))
    print("2nd DTD", sum(top_5_2) / len(top_5_2))
