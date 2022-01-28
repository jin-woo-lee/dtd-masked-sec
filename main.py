#!/usr/bin/env python3
import argparse
import os
from train import train
from test import test
import logging
import logging.handlers

if __name__=='__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    #------------------------------ 
    # General
    #------------------------------ 
    parser.add_argument('--train', type=str2bool, default='false')
    parser.add_argument('--test', type=str2bool, default='false')
    parser.add_argument('--resume', type=str2bool, default='false')
    parser.add_argument('--plot_epoch', type=int, default=20)
    parser.add_argument('--save_epoch', type=int, default=1000)
    parser.add_argument('--test_fold', type=int, default=5)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint')
    parser.add_argument('--test_dir', type=str, default=None, help='directory path of ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of inference')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--exp_name', type=str, default='SpecDTD')
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--vanilla_train', type=str2bool, default='true', help='train on original input')
    parser.add_argument('--masked_train', type=str2bool, default='true', help='train by relative-map masked input')
    parser.add_argument('--force_mask', type=str2bool, default='false', help='force mask to be shaped as a source')
    parser.add_argument('--train_selective', type=str2bool, default='false', help='impose masked_train only if the loss is lower')
    parser.add_argument('--model', type=str, default='vgg16_bn',
                        choices=['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                                 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34',
                                 'resnet50', 'resnet101', 'resnet152']
    )
    parser.add_argument('--sample_dir', type=str, default='saliency_map',
                        help='directory of saliency map heatmap sample')
    #------------------------------ 
    # DDP
    #------------------------------ 
    parser.add_argument('--gpus', nargs='+', default=[0,1], help='gpus')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--port', default='1234', type=str, help='port')
    #------------------------------ 
    # DTD
    #------------------------------ 
    parser.add_argument('--low', type=float, default=0., help='DTD root box lower bound')
    parser.add_argument('--high', type=float, default=1., help='DTD root box upper bound')
    parser.add_argument('--power', type=float, default=0.4)
    parser.add_argument('--gain', type=int, default=40)
    parser.add_argument('--alpha', type=float, default=0.1, help='saliency map regularization coefficient')
    #parser.add_argument('--power', type=float, default=0.3)
    #parser.add_argument('--gain', type=int, default=20)
    #parser.add_argument('--power', type=float, default=0.5)
    #parser.add_argument('--gain', type=int, default=100)
    #------------------------------ 
    # Data
    #------------------------------ 
    parser.add_argument('--task', type=str, default='SEC', choices=['SEC', 'ASR', 'FOR'])
    parser.add_argument('--noisy', type=str2bool, default='false')
    parser.add_argument('--snr', type=float, default=0.)
    parser.add_argument('--framelen', type=int, default=64000)
    parser.add_argument('--classes', type=int, default=251)
    parser.add_argument('--n_mels', type=int, default=251)
    #parser.add_argument('--n_mels', type=int, default=128)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = args.port

    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("INFO")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if args.train:
        if args.resume and args.ckpt==None:
            ckpt_path = f'results/{args.exp_name}/checkpoints/{args.load_epoch}/classifier_{args.load_step}.pt'
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
            else:
                raise FileNotFoundError(
                    "Specify checkpoint by '--ckpt=...'.",
                    "Otherwise provide exact setting for exp_name, load_epoch, load_step.",
                    cpkt_path
                )
        train(args)
    if args.test:
        if args.ckpt==None:
            ckpt_path = f'results/{args.exp_name}/checkpoints/{args.load_epoch}/classifier_{args.load_step}.pt'
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
            else:
                raise FileNotFoundError(
                    f"Specify checkpoint by '--ckpt=...'. "
                    f"Otherwise provide exact setting for exp_name, load_epoch, load_step. {cpkt_path}"
                )
        test(args)


