import os
import time
import torch
from solver import Solver
import torch.multiprocessing as mp
import torch.distributed as dist
import logging

def test(args):
    solver = Solver(args)
    
    ngpus_per_node = int(torch.cuda.device_count()/args.n_nodes)
    logging.info("use {} gpu machine".format(ngpus_per_node))
    args.world_size = ngpus_per_node * args.n_nodes
    mp.spawn(worker, nprocs=ngpus_per_node, args=(solver, ngpus_per_node, args))


def worker(gpu, solver, ngpus_per_node, args):
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', 
                            world_size=args.world_size,
                            init_method='env://',
                            rank=args.rank)
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    solver.set_gpu(args)
    
    if args.rank % ngpus_per_node == 0:
        solver.test(args)
    logging.info('test finished')

