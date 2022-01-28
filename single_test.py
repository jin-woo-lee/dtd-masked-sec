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
from torch.utils.data import RandomSampler
from model import *
from model import saliency_mapping as sa_map
from torch.autograd import Variable
import torch.optim as optim
from utils import rms_normalize, adjust_noise

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

def update_loader(dataset):
    datasampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        #multiprocessing_context=self.mp_context,
        pin_memory=True, sampler=datasampler, drop_last=True,
        worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
    )
    return data_loader

def get_basis(mode,device,n_mels,sr):
    if mode=='inv':
        basis = librosa.filters.mel(sr,1024,n_mels=n_mels,norm=None,fmin=0,fmax=sr//2).T
    else:
        basis = librosa.filters.mel(sr,1024,n_mels=n_mels,fmin=0,fmax=sr//2)
    basis = np.expand_dims(np.expand_dims(basis,axis=0),axis=0)
    return torch.from_numpy(basis).to(device)

def to_melspec(x, win, mel_basis, normalize=True):
    spec = torch.stft(
        x, 1024, hop_length=256, win_length=1024,
        center=True, pad_mode='reflect', window=win
    )
    mag_sp = torch.sqrt(spec.pow(2).sum(-1) + 1e-5)
    phs_sp = spec / mag_sp.unsqueeze(-1).repeat(1,1,1,2)
    mag_sp = torch.matmul(mel_basis, mag_sp).clamp((1e-5)**.5,).squeeze(0) 
    logmel = torch.log(mag_sp).unsqueeze(1)

    if normalize:
        logmel = (logmel - np.log(1e-5)) / (np.log(1024/80) - np.log(1e-5))
    return logmel, phs_sp

def plot_spec(x, path, title):
    x = x.detach().cpu().numpy()
    plt.figure(figsize=(7,7))
    librosa.display.specshow(x, cmap='magma')
    plt.title(title)
    plt.clim(-11,5)
    plt.colorbar()
    plt.savefig(path)
    plt.close()

def test(args):
    
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("INFO")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info('data loading')

    module = __import__('dataset.loader', fromlist=[''])
    data_root = '/data2/ESC-50-master'
    with open(os.path.join(data_root,'meta/esc50.csv'), newline='') as csvfile:
        meta_data = list(csv.reader(csvfile))
    meta_data = np.array(meta_data)[1:]    # (2000,7)
    data_dir = f'{data_root}/audio'
    testset = module.Testset(
        meta_data = meta_data,
        directory=data_dir,
        num_classes=args.num_classes,
        framelen=args.framelen, sr=16000,
    )
    device = f'cuda:{args.gpu}'

    logging.info('prepare model')
    classes = args.num_classes

    if args.model.startswith('vgg'):
        model_archi = 'vgg'
        if args.model.endswith('11'):
            model = vgg11(num_classes=classes,pretrained=False)
        elif args.model.endswith('11_bn'):
            model = vgg11_bn(num_classes=classes,pretrained=False)
        elif args.model.endswith('13'):
            model = vgg13(num_classes=classes,pretrained=False)
        elif args.model.endswith('13_bn'):
            model = vgg13_bn(num_classes=classes,pretrained=False)
        elif args.model.endswith('16'):
            model = vgg16(num_classes=classes,pretrained=False)
        elif args.model.endswith('16_bn'):
            model = vgg16_bn(num_classes=classes,pretrained=False)
        elif args.model.endswith('19'):
            model = vgg19(num_classes=classes,pretrained=False)
        elif args.model.endswith('19_bn'):
            model = vgg19_bn(num_classes=classes,pretrained=False)
    elif args.model.startswith('resnet'):
        model_archi = 'resnet'
        if args.model.endswith('18'):
            model = resnet18(num_classes=classes,pretrained=False)
        elif args.model.endswith('34'):
            model = resnet34(num_classes=classes,pretrained=False)
        elif args.model.endswith('50'):
            model = resnet50(num_classes=classes,pretrained=False)
        elif args.model.endswith('101'):
            model = resnet101(num_classes=classes,pretrained=False)
        elif args.model.endswith('152'):
            model = resnet152(num_classes=classes,pretrained=False)
    else:
        raise ValueError(f"{args.model} is not available")

    print("Load checkpoint from: {}:".format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location=device)
    old_keys = list(checkpoint["model"].keys())
    for keys in old_keys:
        _key = keys.split('module.')[-1]
        checkpoint["model"][_key] = checkpoint["model"].pop(keys)
    model.load_state_dict(checkpoint["model"])
    epoch = (int)(checkpoint["epoch"])
    step  = (int)(checkpoint["step"])

    model.to(device)
    win = torch.hann_window(1024,periodic=True).to(device)

    model.train(False)
    module_list = sa_map.model_flattening(model)
    act_store_model = sa_map.ActivationStoringNet(module_list)
    DTD = sa_map.DTD(lowest=args.low, highest=args.high, device=device)
    loss_func = nn.CrossEntropyLoss()

    test_top1 = 0
    test_top5 = 0
    test_count = 0
    mel_basis = get_basis('mel',device,args.n_mels,16000)
    inv_basis = get_basis('inv',device,args.n_mels,16000)
    test_loader = update_loader(testset)
    with torch.no_grad():
        for i, ts in enumerate(test_loader):
            (mix, speech, noises, target, label) = ts
            length = mix.shape[-1]
            mixture = Variable(mix).to(device).float()
            speech = Variable(speech).to(device).float()
            noises = Variable(noises).to(device).float()
            label = Variable(label).to(device)

            # Prepare data and model
            mix_mel, mix_phs = to_melspec(mixture, win, mel_basis)
            spc_mel, spc_phs = to_melspec(speech, win, mel_basis)
            nos_mel, nos_phs = to_melspec(noises, win, mel_basis)
            module_list = sa_map.model_flattening(model)
            act_store_model.update_module_list(module_list)

            # 1st time DTD
            module_stack, out_1 = act_store_model(mix_mel)
            saliency_1 = DTD(module_stack, out_1, classes, model_archi)
            saliency_1 = args.gain * (saliency_1**args.power) 
            sal_mask_1 = saliency_1.clamp(0,1) * mix_mel
            loss_1 = loss_func(out_1, label)

            # 2nd time DTD
            module_stack, out_2 = act_store_model(sal_mask_1)
            saliency_2 = DTD(module_stack, out_2, classes, model_archi)
            saliency_2 = args.gain * (saliency_2**args.power) 
            sal_mask_2 = saliency_2.clamp(0,1) * mix_mel
            loss_2 = loss_func(out_2, label)

            # 3rd time DTD
            module_stack, out_3 = act_store_model(sal_mask_2)
            saliency_3 = DTD(module_stack, out_3, classes, model_archi)
            saliency_3 = args.gain * (saliency_3**args.power) 
            #sal_mask_3 = saliency_3.clamp(0,1) * mix_mel
            loss_3 = loss_func(out_3, label)
            #------------------------------ 
            binary_3 = torch.zeros_like(saliency_3)
            binary_3[saliency_3 > 0.60] = 0.2
            binary_3[saliency_3 > 0.65] = 0.5
            binary_3[saliency_3 > 0.70] = 1.0
            sal_mask_3 = binary_3 * mix_mel
            #------------------------------ 

            losses_1 = torch.mean(loss_1).data
            losses_2 = torch.mean(loss_2).data
            losses_3 = torch.mean(loss_3).data
            logging.info((f"Test, epoch #{epoch}, step #{i}, "
                          f"original loss {losses_1:.3f}, "
                          f"masked-2 loss {losses_2:.3f}, "
                          f"masked-3 loss {losses_3:.3f}, "))

            acc1, acc5 = accuracy(out_1, label, topk=(1, 5))
            test_count += mix_mel.size(0)
            test_top1 += acc1[0].sum() * mix_mel.size(0)
            test_top1_avg = test_top1 / test_count
            test_top5 += acc5[0].sum() * mix_mel.size(0)
            test_top5_avg = test_top5 / test_count

            #------------------------------ 
            # Plot
            #------------------------------ 
            out_1 = out_1.detach().cpu().numpy()
            out_2 = out_2.detach().cpu().numpy()
            out_3 = out_3.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            salmsk_1 = torch.matmul(inv_basis, sal_mask_1)
            salmap_1 = torch.matmul(inv_basis, saliency_1)
            salmsk_2 = torch.matmul(inv_basis, sal_mask_2)
            salmap_2 = torch.matmul(inv_basis, saliency_2)
            salmsk_3 = torch.matmul(inv_basis, sal_mask_3)
            salmap_3 = torch.matmul(inv_basis, saliency_3)
            mix_mel = torch.matmul(inv_basis, mix_mel)
            spc_mel = torch.matmul(inv_basis, spc_mel)
            nos_mel = torch.matmul(inv_basis, nos_mel)
            logging.info('sample saliency map generation')
            saliency_dir = os.path.join('./results/{}/test'.format(args.exp_name), str(epoch)+'-'+str(step))
            os.makedirs(saliency_dir, exist_ok=True)
            bz = label.shape[0]
            for j in range(bz):
                sal_map_1 = salmap_1[j].squeeze()
                sal_msk_1 = salmsk_1[j].squeeze()
                sal_map_2 = salmap_2[j].squeeze()
                sal_msk_2 = salmsk_2[j].squeeze()
                sal_map_3 = salmap_3[j].squeeze()
                sal_msk_3 = salmsk_3[j].squeeze()
                sample_origin = mix_mel[j].squeeze()
                speech_origin = spc_mel[j].squeeze()
                noises_origin = nos_mel[j].squeeze()
                sal_map_1 = sal_map_1 * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                sal_msk_1 = sal_msk_1 * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                sal_map_2 = sal_map_2 * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                sal_msk_2 = sal_msk_2 * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                sal_map_3 = sal_map_3 * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                sal_msk_3 = sal_msk_3 * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                sample_origin = sample_origin * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                speech_origin = speech_origin * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                noises_origin = noises_origin * (np.log(1024/80) - np.log(1e-5)) + np.log(1e-5)
                s1 = sal_msk_1.unsqueeze(-1).repeat(1,1,2)
                s2 = sal_msk_2.unsqueeze(-1).repeat(1,1,2)
                s3 = sal_msk_3.unsqueeze(-1).repeat(1,1,2)
                so = sample_origin.unsqueeze(-1).repeat(1,1,2)
                sp = mix_phs[j].squeeze()

                sali_sp_1 = torch.exp(s1) * sp
                sali_sp_2 = torch.exp(s2) * sp
                sali_sp_3 = torch.exp(s3) * sp
                samp_spec = torch.exp(so) * sp
                sali_w_1 = torch.istft(sali_sp_1, 1024, hop_length=256, win_length=1024, center=True, length=length)
                sali_w_2 = torch.istft(sali_sp_2, 1024, hop_length=256, win_length=1024, center=True, length=length)
                sali_w_3 = torch.istft(sali_sp_3, 1024, hop_length=256, win_length=1024, center=True, length=length)
                samp_wav = torch.istft(samp_spec, 1024, hop_length=256, win_length=1024, center=True, length=length)
                sali_w_1 = sali_w_1.cpu().numpy()
                sali_w_2 = sali_w_2.cpu().numpy()
                sali_w_3 = sali_w_3.cpu().numpy()
                samp_wav = samp_wav.cpu().numpy()
                sali_w_1 = rms_normalize(sali_w_1)
                sali_w_2 = rms_normalize(sali_w_2)
                sali_w_3 = rms_normalize(sali_w_3)
                samp_wav = rms_normalize(samp_wav)
                s_w_1_path = os.path.join(saliency_dir, f"{i}-{j}th_wav_1_saliency.wav")
                s_w_2_path = os.path.join(saliency_dir, f"{i}-{j}th_wav_2_saliency.wav")
                s_w_3_path = os.path.join(saliency_dir, f"{i}-{j}th_wav_3_saliency.wav")
                o_wav_path = os.path.join(saliency_dir, f"{i}-{j}th_wav_0_original.wav")
                sf.write(s_w_1_path, sali_w_1, samplerate=16000, subtype='PCM_16')
                sf.write(s_w_2_path, sali_w_2, samplerate=16000, subtype='PCM_16')
                sf.write(s_w_3_path, sali_w_3, samplerate=16000, subtype='PCM_16')
                sf.write(o_wav_path, samp_wav, samplerate=16000, subtype='PCM_16')

                #------------------------------ 
                logit_1 = out_1[j,label[j]]
                logit_2 = out_2[j,label[j]]
                logit_3 = out_3[j,label[j]]
                objct = category[j]
                title = f'{objct}: {logit_1}'
                tit_2 = f'{objct}: {logit_2}'
                tit_3 = f'{objct}: {logit_3}'
                plot_spec(sal_map_1, os.path.join(saliency_dir, f"{i}-{j}th_spec_1_salmap.png"), title)
                plot_spec(sal_msk_1, os.path.join(saliency_dir, f"{i}-{j}th_spec_1_sample.png"), title)
                plot_spec(sal_map_2, os.path.join(saliency_dir, f"{i}-{j}th_spec_2_salmap.png"), tit_2)
                plot_spec(sal_msk_2, os.path.join(saliency_dir, f"{i}-{j}th_spec_2_sample.png"), tit_2)
                plot_spec(sal_map_3, os.path.join(saliency_dir, f"{i}-{j}th_spec_3_salmap.png"), tit_3)
                plot_spec(sal_msk_3, os.path.join(saliency_dir, f"{i}-{j}th_spec_3_sample.png"), tit_3)
                plot_spec(sample_origin, os.path.join(saliency_dir, f"{i}-{j}th_spec_origin.png"), title)
                plot_spec(speech_origin, os.path.join(saliency_dir, f"{i}-{j}th_spec_speech.png"), title)
                plot_spec(noises_origin, os.path.join(saliency_dir, f"{i}-{j}th_spec_noise.png"), title)
            #------------------------------ 

    logging.info('test finish')
    logging.info(log)

if __name__=='__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str2bool, default='true')
    parser.add_argument('--test', type=str2bool, default='true')
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint')
    parser.add_argument('--low', type=float, default=0., help='DTD root box lower bound')
    parser.add_argument('--high', type=float, default=1., help='DTD root box upper bound')
    parser.add_argument('--gpu', type=int, default=12, help='gpu')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of inference')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--power', type=float, default=0.4)
    parser.add_argument('--gain', type=int, default=50)
    #parser.add_argument('--power', type=float, default=0.3)
    #parser.add_argument('--gain', type=int, default=20)
    #parser.add_argument('--power', type=float, default=0.5)
    #parser.add_argument('--gain', type=int, default=100)
    parser.add_argument('--framelen', type=int, default=64000)
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--dtd_every', type=int, default=15)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=1)
    parser.add_argument('--n_mels', type=int, default=251)
    parser.add_argument('--exp_name', type=str, default='SpecDTD')
    parser.add_argument('--task', type=str, default='detect',
                        choices=['recog', 'detect']
    )
    parser.add_argument('--model', type=str, default='vgg16_bn',
                        choices=['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                                 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34',
                                 'resnet50', 'resnet101', 'resnet152']
    )
    args = parser.parse_args()

    if args.ckpt==None:
        ckpt_path = f'results/{args.exp_name}/checkpoints/{args.load_epoch}/classifier_{args.load_step}.pt'
        if os.path.exists(ckpt_path):
            args.ckpt = ckpt_path
        else:
            raise FileNotFoundError(
                f"Specify checkpoint by '--ckpt=...'. "
                f"Otherwise provide exact setting for exp_name, load_epoch, load_step. {ckpt_path}",
            )
    test(args)
