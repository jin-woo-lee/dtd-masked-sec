#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os
import random
import csv
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchaudio import transforms
from torchaudio import datasets
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from model import *
from model import saliency_mapping as sa_map
from utils import *
from torch.autograd import Variable
import torch.optim as optim

def accuracy(output, label, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1,batch_size).float().sum(0, keepdim=True)
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Solver(object):
    def __init__(self, args):

        logging.info('Initialize solver with dataset')
        module = __import__('dataset.loader', fromlist=[''])
        #============================== 
        # SEC
        #============================== 
        self.mp_context = torch.multiprocessing.get_context('fork')
        if args.task=='SEC':
            self.set_sec_dataset(args, module)
        elif args.task=='ASR':
            self.set_asr_dataset(args, module)
        elif args.task=='FOR':
            self.set_for_dataset(args, module)
        else:
            raise NotImplementedError("Undefined task")

    def set_gpu(self, args):
        logging.info('set distributed data parallel')
        if args.train:
            self.train_sampler = DistributedSampler(self.trainset,shuffle=True,rank=args.gpu,seed=0)
            self.train_loader = DataLoader(
                self.trainset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.train_sampler, drop_last=True,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
            )
        if args.test:
            self.test_sampler = DistributedSampler(self.testset,shuffle=False,rank=args.gpu)
            self.test_loader = DataLoader(
                self.testset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.test_sampler, drop_last=True,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
            )

        logging.info('set device for model')
        self.num_classes = args.num_classes
    
        if args.model.startswith('vgg'):
            self.model_archi = 'vgg'
            if args.model.endswith('11'):
                model = vgg11(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('11_bn'):
                model = vgg11_bn(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('13'):
                model = vgg13(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('13_bn'):
                model = vgg13_bn(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('16'):
                model = vgg16(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('16_bn'):
                model = vgg16_bn(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('19'):
                model = vgg19(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('19_bn'):
                model = vgg19_bn(num_classes=self.num_classes,pretrained=False)
        elif args.model.startswith('resnet'):
            self.model_archi = 'resnet'
            if args.model.endswith('18'):
                model = resnet18(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('34'):
                model = resnet34(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('50'):
                model = resnet50(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('101'):
                model = resnet101(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('152'):
                model = resnet152(num_classes=self.num_classes,pretrained=False)
        #------------------------------ 
        elif args.model.startswith('unet'):
            self.model_archi = 'unet'
            if args.model.endswith('5'):
                model = unet5(num_classes=self.num_classes,pretrained=False)
            elif args.model.endswith('5_bn'):
                model = unet5_bn(num_classes=self.num_classes,pretrained=False)
        #------------------------------ 
        else:
            raise ValueError(f"{args.model} is not available")
        self.act_store_model = sa_map.ActivationStoringNet(None)
        self.act_store_model.to(args.gpu)
        self.DTD = sa_map.DTD(lowest=args.low, highest=args.high, device=args.gpu)
        self.fwd_criterion = nn.CrossEntropyLoss()
        self.bwd_criterion = nn.L1Loss()
        self.reg_criterion = nn.MSELoss()
    
        self.optimizer = torch.optim.Adam(
            params=list(model.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
    
        self.win = torch.hann_window(1024,periodic=True).to(args.gpu)
        self.mel_basis = get_basis('mel',args.gpu,args.n_mels,44100)
        self.inv_basis = get_basis('inv',args.gpu,args.n_mels,44100)
        torch.cuda.set_device(args.gpu)

        # Distribute models to machine
        model = model.to('cuda:{}'.format(args.gpu))
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.gpu], output_device=args.gpu, find_unused_parameters=True) 
        self.model = ddp_model

        if args.resume or args.test:
            print("Load checkpoint from: {}:".format(args.ckpt))
            checkpoint = torch.load(args.ckpt, map_location=f'cuda:{args.gpu}')
            self.model.load_state_dict(checkpoint["model"])
            self.start_epoch = (int)(checkpoint["epoch"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.start_epoch = 0

    def set_sec_dataset(self, args, module):
        data_root = '/data2/ESC-50-master'
        with open(os.path.join(data_root,'meta/esc50.csv'), newline='') as csvfile:
            meta_data = list(csv.reader(csvfile))
        # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
        meta_data = np.array(meta_data)[1:]    # (2000,7)

        if args.train:
            train_s_dir = f'{data_root}/audio'
            self.trainset = module.Trainset(
                meta_data = meta_data,
                directory=train_s_dir,
                num_classes=args.num_classes,
                framelen=args.framelen, sr=44100,
                noisy = args.noisy,
                test_fold=args.test_fold,
            )
        if args.test:
            test_s_dir = f'{data_root}/audio'
            self.testset = module.Testset(
                meta_data = meta_data,
                directory=test_s_dir,
                num_classes=args.num_classes,
                framelen=args.framelen, sr=44100,
                noisy = args.noisy,
                snr = args.snr,
                test_fold=args.test_fold,
            )

    def set_asr_dataset(self, args, module):
        data_root = '/data/LibriSpeech'

        if args.train:
            train_s_dir = f'{data_root}/train-clean-500'
            self.trainset = module.Trainset(
                meta_data = meta_data,
                directory=train_s_dir,
                num_classes=args.num_classes,
                framelen=args.framelen, sr=44100,
                noisy = args.noisy,
                test_fold=args.test_fold,
            )
        if args.test:
            test_s_dir = f'{data_root}/test-clean'
            self.testset = module.Testset(
                meta_data = meta_data,
                directory=test_s_dir,
                num_classes=args.num_classes,
                framelen=args.framelen, sr=44100,
                noisy = args.noisy,
                snr = args.snr,
                test_fold=args.test_fold,
            )

    def set_for_dataset(self, args, module):
        data_root = '/data2/ESC-50-master'
        with open(os.path.join(data_root,'meta/esc50.csv'), newline='') as csvfile:
            meta_data = list(csv.reader(csvfile))
        # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
        meta_data = np.array(meta_data)[1:]    # (2000,7)

        if args.train:
            train_s_dir = f'{data_root}/audio'
            self.trainset = module.Trainset(
                meta_data = meta_data,
                directory=train_s_dir,
                num_classes=args.num_classes,
                framelen=args.framelen, sr=44100,
                noisy = args.noisy,
                test_fold=args.test_fold,
            )
        if args.test:
            test_s_dir = f'{data_root}/audio'
            self.testset = module.Testset(
                meta_data = meta_data,
                directory=test_s_dir,
                num_classes=args.num_classes,
                framelen=args.framelen, sr=44100,
                noisy = args.noisy,
                snr = args.snr,
                test_fold=args.test_fold,
            )

    def save_checkpoint(self, args, epoch, step, checkpoint_dir, name):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step" : step}
        checkpoint_path = os.path.join(checkpoint_dir,'{}_{}.pt'.format(name, step))
        torch.save(checkpoint_state, checkpoint_path)
        print("Saved checkpoint: {}".format(checkpoint_path))
    
    def train(self, args):
    
        self.model.train()
    
        logging.info('train start')
        for epoch in range(self.start_epoch, args.total_epochs+1):
            for i, ts in enumerate(self.train_loader):
                (audios, speech, labels, category) = ts
                length = audios.shape[-1]

                audios = Variable(audios).to(args.gpu).float()
                speech = Variable(speech).to(args.gpu).float()
                labels = Variable(labels).to(args.gpu)
                mix_mel, phs = to_melspec(audios, self.win, self.mel_basis)
                tar_mel, ___ = to_melspec(speech, self.win, self.mel_basis)
                loss = 0
    
                #============================== 
                # Vanilla Train
                #============================== 
                output_1 = self.model(mix_mel)
                loss_1 = self.fwd_criterion(output_1, labels)
                if args.vanilla_train:
                    self.optimizer.zero_grad()
                    loss_1.backward()
                    self.optimizer.step()
                    loss += loss_1
    
                #============================== 
                # Masked Train
                #============================== 
                if args.force_mask:
                    #============================== 
                    # Backward Train
                    #============================== 
                    module_list = sa_map.model_flattening(self.model)
                    self.act_store_model.update_module_list(module_list)
                    #------------------------------ 
                    #module_stack, output = self.act_store_model(mix_mel)
                    #saliency_map = self.DTD(module_stack, output, self.num_classes, self.model_archi)
                    #------------------------------ 
                    module_stack, output = self.act_store_model(tar_mel)
                    l_tensor = torch.zeros_like(output)+1e-20
                    for b in range(len(output)):
                        l_tensor[b,labels[b]] = 1e20
                    l_tensor = l_tensor.log()
                    saliency_map = self.DTD(module_stack, l_tensor, self.num_classes, self.model_archi)
                    #------------------------------ 

                    saliency_map = args.gain * (saliency_map**args.power) 
                    sal_masked = saliency_map.clamp(0,1) * mix_mel
                    sm_normed = minmax_normalize(sal_masked)
                    tm_normed = minmax_normalize(tar_mel)
                    #------------------------------ 
                    mel_loss = self.bwd_criterion(sm_normed, tm_normed)
                    #------------------------------ 
                    #lower_bd = 0
                    ##lower_bd = -0.1
                    #undesired = (sm_normed - tm_normed).clamp(lower_bd,)
                    #zeros_mel = torch.zeros_like(undesired)
                    #mel_loss = self.bwd_criterion(undesired, zeros_mel)
                    #------------------------------ 
                    loss_b = 0.01 * mel_loss
                    #loss_b = mel_loss + 0.001 * entropy_reg
                    self.optimizer.zero_grad()
                    loss_b.backward()
                    self.optimizer.step()
    
                else:
                    with torch.no_grad():
                        module_list = sa_map.model_flattening(self.model)
                        self.act_store_model.update_module_list(module_list)
                        module_stack, output = self.act_store_model(mix_mel)
                        saliency_map = self.DTD(module_stack, output, self.num_classes, self.model_archi)
                    saliency_map = args.gain * (saliency_map**args.power) 
                    sal_masked = saliency_map.clamp(0,1) * mix_mel

                if args.masked_train:
                    do_masked_train = False
                    #------------------------------ 
                    # TODO: ?
                    if args.vanilla_train:
                        sal_masked = sal_masked.detach()
                    #------------------------------ 
                    output_2 = self.model(sal_masked)
                    loss_2 = self.fwd_criterion(output_2, labels)
                    if args.train_selective:
                        if loss_2 < loss_1:
                            do_masked_train = True
                    else:
                        do_masked_train = True
                    if do_masked_train:
                        self.optimizer.zero_grad()
                        loss_2.backward()
                        self.optimizer.step()
                    loss += loss_2
    
                if args.gpu==0:
                    if epoch % args.plot_epoch == 0:
                        sal_masked = sal_masked.detach()
                        saliency_map = saliency_map.detach()
                        sal_msk = torch.matmul(self.inv_basis, sal_masked)
                        sal_map = torch.matmul(self.inv_basis, saliency_map)
                        spec_in = torch.matmul(self.inv_basis, mix_mel)
                        #logging.info('sample saliency map generation')
                        saliency_dir = os.path.join('./results/{}/saliency_map'.format(args.exp_name), str(epoch))
                        os.makedirs(saliency_dir, exist_ok=True)
                        saliency_maps = sal_map[0].squeeze()
                        saliency_mskd = sal_msk[0].squeeze()
                        sample_origin = spec_in[0].squeeze()
                        saliency_maps = saliency_maps * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                        saliency_mskd = saliency_mskd * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                        sample_origin = sample_origin * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                        ss = saliency_mskd.unsqueeze(-1).repeat(1,1,2)
                        sm = saliency_maps.unsqueeze(-1).repeat(1,1,2)
                        so = sample_origin.unsqueeze(-1).repeat(1,1,2)
                        sp = phs[0].squeeze()
    
                        sali_spec = torch.exp(ss) * sp
                        samp_spec = torch.exp(so) * sp
                        sali_wav = torch.istft(sali_spec, 1024, hop_length=256, win_length=1024, center=True, length=length)
                        samp_wav = torch.istft(samp_spec, 1024, hop_length=256, win_length=1024, center=True, length=length)
                        sali_wav = sali_wav.cpu().numpy()
                        samp_wav = samp_wav.cpu().numpy()
                        sali_wav = rms_normalize(sali_wav)
                        samp_wav = rms_normalize(samp_wav)
                        s_wav_path = os.path.join(saliency_dir, f"{i}th_saliency.wav")
                        o_wav_path = os.path.join(saliency_dir, f"{i}th_original.wav")
                        sf.write(s_wav_path, sali_wav, samplerate=44100,subtype='PCM_16')
                        sf.write(o_wav_path, samp_wav, samplerate=44100,subtype='PCM_16')
    
                        #------------------------------ 
                        output = output_1.detach().cpu().numpy()
                        labels = labels.detach().cpu().numpy()
                        logit = output[0,labels[0]]
                        objct = category[0]
                        title = f'{objct}: {logit}'
                        plot_spec(saliency_maps, os.path.join(saliency_dir, f"{i}th_spec_salmap.png"), title)
                        plot_spec(saliency_mskd, os.path.join(saliency_dir, f"{i}th_spec_sample.png"), title)
                        plot_spec(sample_origin, os.path.join(saliency_dir, f"{i}th_spec_origin.png"), title)
                        #------------------------------ 
    
                    if epoch % args.save_epoch == 0 and i==1:
                        checkpoint_dir = os.path.join('./results/{}/checkpoints'.format(args.exp_name), str(epoch))
                        self.save_checkpoint(args, epoch, i, checkpoint_dir, 'classifier')
    
                    if i % 5 == 0:
                        loss_ = torch.mean(loss).data
                        #logging.info((f"Train, epoch #{epoch}/{args.total_epochs},"
                        #              f"step #{i}/{len(self.train_loader)},, "
                        #              #f"top1 accuracy {test_top1_avg.data:.3f}, "
                        #              #f"top5 accuracy {test_top5_avg.data:.3f}, "
                        #              f"loss {loss_:.3f}, "))
                        print(f"Train, epoch #{epoch}/{args.total_epochs},\t"
                              f"step #{i}/{len(self.train_loader)},\t"
                              f"loss {loss_:.3f},\t")
    
    def test(self, args):

        test_top1 = 0
        test_top5 = 0
        test_count = 0
        self.mel_basis = get_basis('mel',args.gpu,args.n_mels,44100)
        self.inv_basis = get_basis('inv',args.gpu,args.n_mels,44100)
        self.model.eval()
        #logging.info('test start')
        if args.gpu==0:
            print('test start')
        log = []
        with torch.no_grad():
            for i, ts in enumerate(self.test_loader):
                (audios, speech, labels, category) = ts
    
                # Prepare data and model
                length = audios.shape[-1]
                audios = Variable(audios).to(args.gpu).float()
                speech = Variable(speech).to(args.gpu).float()
                labels = Variable(labels).to(args.gpu)
                mix_mel, phs = to_melspec(audios, self.win, self.mel_basis)
                tar_mel, ___ = to_melspec(speech, self.win, self.mel_basis)

                module_list = sa_map.model_flattening(self.model)
                self.act_store_model.update_module_list(module_list)
    
                # 1st time DTD
                module_stack, out_1 = self.act_store_model(mix_mel)
                saliency_1 = self.DTD(module_stack, out_1, self.num_classes, self.model_archi)
                saliency_1 = args.gain * (saliency_1**args.power) 
                sal_mask_1 = saliency_1.clamp(0,1) * mix_mel
                loss_1 = self.fwd_criterion(out_1, labels)
    
                # 2nd time DTD
                module_stack, out_2 = self.act_store_model(sal_mask_1)
                saliency_2 = self.DTD(module_stack, out_2, self.num_classes, self.model_archi)
                saliency_2 = args.gain * (saliency_2**args.power) 
                sal_mask_2 = saliency_2.clamp(0,1) * mix_mel
                loss_2 = self.fwd_criterion(out_2, labels)
    
                # 3rd time DTD
                #module_stack, out_3 = self.act_store_model(sal_mask_2)
                #saliency_3 = self.DTD(module_stack, out_3, self.num_classes, self.model_archi)
                #saliency_3 = args.gain * (saliency_3**args.power) 
                #sal_mask_3 = saliency_3.clamp(0,1) * mix_mel
                #loss_3 = self.fwd_criterion(out_3, labels)
                #------------------------------ 
                module_stack, out_3 = self.act_store_model(mix_mel)
                lab_o = torch.zeros_like(out_2) + 1e-20
                for b in range(len(out_2)):
                    lab_o[b,labels[b]] = 1e20
                lab_o = lab_o.log()
                saliency_3 = self.DTD(module_stack, lab_o, self.num_classes, self.model_archi)
                saliency_3 = args.gain * (saliency_3**args.power) 
                sal_mask_3 = saliency_3.clamp(0,1) * mix_mel
                loss_3 = self.fwd_criterion(out_1, labels)
                #------------------------------ 
    
                losses_1 = torch.mean(loss_1).data
                losses_2 = torch.mean(loss_2).data
                losses_3 = torch.mean(loss_3).data
                if args.gpu==0:
                    print(f"Test set {i}/{len(self.test_loader)},\t"
                          f"epoch #{args.load_epoch}, step #{args.load_step},\t"
                          f"original loss {losses_1:.3f},\t"
                          f"masked-2 loss {losses_2:.3f},\t"
                          f"masked-3 loss {losses_3:.3f},\t")
    
                # 1st time Accuracy
                acc1, acc5 = accuracy(out_1, labels, topk=(1, 5))
                test_count += mix_mel.size(0)
                test_top1 += acc1[0].sum() * mix_mel.size(0)
                #test_top1 += acc1[0] * mix_mel.size(0)
                test_top1_avg = test_top1 / test_count
                test_top5 += acc5[0].sum() * mix_mel.size(0)
                #test_top5 += acc5[0] * mix_mel.size(0)
                test_top5_avg = test_top5 / test_count
                log.append('\n'.join([
                    f'1st time DTD Test Result',
                    f'- top1 acc: {test_top1_avg:.4f}',
                    f'- top5 acc: {test_top5_avg:.4f}']))
    
                # 2nd time Accuracy
                acc1, acc5 = accuracy(out_2, labels, topk=(1, 5))
                test_count += mix_mel.size(0)
                test_top1 += acc1[0].sum() * mix_mel.size(0)
                test_top1_avg = test_top1 / test_count
                test_top5 += acc5[0].sum() * mix_mel.size(0)
                test_top5_avg = test_top5 / test_count
                log.append('\n'.join([
                    f'2nd time DTD Test Result',
                    f'- top1 acc: {test_top1_avg:.4f}',
                    f'- top5 acc: {test_top5_avg:.4f}']))
    
                # 3rd time Accuracy
                acc1, acc5 = accuracy(out_3, labels, topk=(1, 5))
                test_count += mix_mel.size(0)
                test_top1 += acc1[0].sum() * mix_mel.size(0)
                test_top1_avg = test_top1 / test_count
                test_top5 += acc5[0].sum() * mix_mel.size(0)
                test_top5_avg = test_top5 / test_count
                log.append('\n'.join([
                    f'3rd time DTD Test Result',
                    f'- top1 acc: {test_top1_avg:.4f}',
                    f'- top5 acc: {test_top5_avg:.4f}']))
    
                #------------------------------ 
                # Plot
                #------------------------------ 
                out_1 = out_1.detach().cpu().numpy()
                out_2 = out_2.detach().cpu().numpy()
                out_3 = out_3.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                salmsk_1 = torch.matmul(self.inv_basis, sal_mask_1)
                salmap_1 = torch.matmul(self.inv_basis, saliency_1)
                salmsk_2 = torch.matmul(self.inv_basis, sal_mask_2)
                salmap_2 = torch.matmul(self.inv_basis, saliency_2)
                salmsk_3 = torch.matmul(self.inv_basis, sal_mask_3)
                salmap_3 = torch.matmul(self.inv_basis, saliency_3)
                smp_spec = torch.matmul(self.inv_basis, mix_mel)
                logging.info('sample saliency map generation')
                flag = 'noisy' if args.noisy else 'clean'
                saliency_dir = os.path.join(f'./results/{args.exp_name}/test-{flag}', str(args.load_epoch)+'-'+str(args.load_step))
                os.makedirs(saliency_dir, exist_ok=True)
                bz = labels.shape[0]
                for j in range(bz):
                    sal_map_1 = salmap_1[j].squeeze()
                    sal_msk_1 = salmsk_1[j].squeeze()
                    sal_map_2 = salmap_2[j].squeeze()
                    sal_msk_2 = salmsk_2[j].squeeze()
                    sal_map_3 = salmap_3[j].squeeze()
                    sal_msk_3 = salmsk_3[j].squeeze()
                    smp_spec_ = smp_spec[j].squeeze()
                    sal_map_1 = sal_map_1*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    sal_msk_1 = sal_msk_1*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    sal_map_2 = sal_map_2*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    sal_msk_2 = sal_msk_2*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    sal_map_3 = sal_map_3*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    sal_msk_3 = sal_msk_3*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    smp_spec_ = smp_spec_*(np.log(1024/80)-np.log(1e-5)) + np.log(1e-5)
                    s1 = sal_msk_1.unsqueeze(-1).repeat(1,1,2)
                    s2 = sal_msk_2.unsqueeze(-1).repeat(1,1,2)
                    s3 = sal_msk_3.unsqueeze(-1).repeat(1,1,2)
                    so = smp_spec_.unsqueeze(-1).repeat(1,1,2)
                    sp = phs[j].squeeze()
    
                    sali_sp_1 = torch.exp(s1) * sp
                    sali_sp_2 = torch.exp(s2) * sp
                    sali_sp_3 = torch.exp(s3) * sp
                    samp_sp_o = torch.exp(so) * sp
                    sali_w_1 = torch.istft(sali_sp_1, 1024, hop_length=256, win_length=1024, center=True, length=length)
                    sali_w_2 = torch.istft(sali_sp_2, 1024, hop_length=256, win_length=1024, center=True, length=length)
                    sali_w_3 = torch.istft(sali_sp_3, 1024, hop_length=256, win_length=1024, center=True, length=length)
                    samp_wav = torch.istft(samp_sp_o, 1024, hop_length=256, win_length=1024, center=True, length=length)
                    sali_w_1 = sali_w_1.cpu().numpy()
                    sali_w_2 = sali_w_2.cpu().numpy()
                    sali_w_3 = sali_w_3.cpu().numpy()
                    samp_wav = samp_wav.cpu().numpy()
                    sali_w_1 = rms_normalize(sali_w_1)
                    sali_w_2 = rms_normalize(sali_w_2)
                    sali_w_3 = rms_normalize(sali_w_3)
                    samp_wav = rms_normalize(samp_wav)
                    s_w_1_path = os.path.join(saliency_dir, f"{i}-{j}-1-w-sal.wav")
                    s_w_2_path = os.path.join(saliency_dir, f"{i}-{j}-2-w-sal.wav")
                    s_w_3_path = os.path.join(saliency_dir, f"{i}-{j}-3-w-sal.wav")
                    o_wav_path = os.path.join(saliency_dir, f"{i}-{j}-0-w-ori.wav")
                    sf.write(s_w_1_path, sali_w_1, samplerate=44100, subtype='PCM_16')
                    sf.write(s_w_2_path, sali_w_2, samplerate=44100, subtype='PCM_16')
                    sf.write(s_w_3_path, sali_w_3, samplerate=44100, subtype='PCM_16')
                    sf.write(o_wav_path, samp_wav, samplerate=44100, subtype='PCM_16')
    
                    #------------------------------ 
                    logit_1 = out_1[j,labels[j]]
                    logit_2 = out_2[j,labels[j]]
                    logit_3 = out_3[j,labels[j]]
                    objct = category[j]
                    title = f'{objct}: {logit_1}'
                    tit_2 = f'{objct}: {logit_2}'
                    tit_3 = f'{objct}: {logit_3}'
                    plot_spec(sal_map_1, os.path.join(saliency_dir, f"{i}-{j}-1-salmap.png"), title)
                    plot_spec(sal_msk_1, os.path.join(saliency_dir, f"{i}-{j}-1-sample.png"), title)
                    plot_spec(sal_map_2, os.path.join(saliency_dir, f"{i}-{j}-2-salmap.png"), tit_2)
                    plot_spec(sal_msk_2, os.path.join(saliency_dir, f"{i}-{j}-2-sample.png"), tit_2)
                    plot_spec(sal_map_3, os.path.join(saliency_dir, f"{i}-{j}-3-salmap.png"), tit_3)
                    plot_spec(sal_msk_3, os.path.join(saliency_dir, f"{i}-{j}-3-sample.png"), tit_3)
                    plot_spec(smp_spec_, os.path.join(saliency_dir, f"{i}-{j}-0-origin.png"), title)
                #------------------------------ 
    
        #logging.info('test finish')
        if args.gpu==0:
            print('test finish')
            flag = 'noisy' if args.noisy else 'clean'
            log_path = f'./results/{args.exp_name}/test-{flag}-log.txt'
            with open(log_path, 'w') as f:
                for ll in log:
                    lls = ll.split('\n')
                    for l in lls:
                        f.write(l)
                        f.write('\n')
            print('saved accuracy log:', log_path)
        #logging.info(log)

