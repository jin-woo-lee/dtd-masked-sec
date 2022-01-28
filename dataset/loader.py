import torch
import numpy as np 
from torch.utils import data
from torch.utils.data import DataLoader
import os 
import soundfile as sf 
import pickle
import glob
import scipy
import random
import librosa
from utils import cal_rms, adjust_noise, rms_normalize
import logging

class Trainset(torch.utils.data.Dataset):

    def __init__(self,
                 meta_data, directory,
                 framelen=32000, sr=16000, num_classes=50, noisy=False,
                 data_per_fold=400, k_folds=5, test_fold=5):
        
        #np.random.seed(0)
        self.meta_data = meta_data
        self.dir = directory
        self.framelen = framelen
        self.noisy=noisy
        self.sr = sr

        logging.info(f"load dataset: {self.dir}")
        self.train_folds = [(k % k_folds)+1 for k in range(test_fold, test_fold+k_folds-1)]
        self.train_files = []
        for f in self.train_folds:
            for i in range(data_per_fold):
                data_num = data_per_fold * (f-1) + i
                self.train_files.append(data_num)
        # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
        # ['1-100032-A-0.wav', '1', '0', 'dog', 'True', '100032', 'A']

        self.noise = []
        self.n_noi = 0
        if self.noisy:
            self.n_dir = f'/data/DEMAND/DEMANDTestForCustomSet(DEMAND_train)'
            snr_min = -10
            snr_max = 20
            snr_nsteps = 50
            self.snr_list = list(np.linspace(snr_min, snr_max, snr_nsteps))
            noise_set = [os.path.join(self.n_dir,d) for d in sorted(os.listdir(self.n_dir)) if os.path.isdir(os.path.join(self.n_dir,d))]
            for ns in noise_set:
                files = sorted(glob.glob(os.path.join(ns,'*.wav')))
                for f in files:
                    x, sr = sf.read(f)
                    if sr != self.sr:
                        x = librosa.resample(x, sr, self.sr)
                    self.noise.append(x)
                    self.n_noi += 1
                    self.noise_len = len(x)
        self.noise = np.array(self.noise)

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):

        data_idx = self.train_files[index]
        name = self.meta_data[data_idx][0]
        target = (int)(self.meta_data[data_idx][2])
        category = self.meta_data[data_idx][3]
        qpath = os.path.join(self.dir,name)
        query = self.load_segment(qpath)
        if self.noisy:
            noise = self.sample_noise()
            snr = random.sample(self.snr_list,1)[0]
            mixdown, query, noise = adjust_noise(noise, query, snr)
        else:
            query = rms_normalize(query)
            mixdown = query

        return (mixdown, query, target, category)

    def load_segment(self, path):
        full, sr = sf.read(path)
        pw = self.framelen - len(full)
        if len(full) < self.framelen:
            full = np.pad(full, pw, 'constant')
        rms = 0
        while rms < 1e-3:
            trim = np.random.randint(len(full) - self.framelen)
            seg = full[trim:trim+self.framelen]
            rms = cal_rms(seg)
        return seg

    def sample_noise(self):
        rs = np.random.randint(self.n_noi)
        rp = np.random.randint(self.noise_len - self.framelen)
        return self.noise[rs,rp:rp+self.framelen]

class Testset(torch.utils.data.Dataset):

    def __init__(self,
                 meta_data, directory,
                 framelen=32000, sr=16000, num_classes=50, noisy=False, snr=0,
                 data_per_fold=400, k_folds=5, test_fold=5):
        
        #np.random.seed(0)
        self.meta_data = meta_data
        self.dir = directory
        self.framelen = framelen
        self.sr = sr
        self.noisy=noisy
        self.snr=snr

        logging.info(f"load dataset: {self.dir}")
        self.test_folds = [test_fold]
        self.test_files = [data_per_fold * (test_fold-1) + i for i in range(data_per_fold)]
        # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
        # ['1-100032-A-0.wav', '1', '0', 'dog', 'True', '100032', 'A']

        self.noise = []
        self.n_noi = 0
        if self.noisy:
            self.n_dir = f'/data/DEMAND/OfficialDEMANDTestSet'
            noise_set = [os.path.join(self.n_dir,d) for d in sorted(os.listdir(self.n_dir)) if os.path.isdir(os.path.join(self.n_dir,d))]
            for ns in noise_set:
                files = sorted(glob.glob(os.path.join(ns,'*.wav')))
                for f in files:
                    x, sr = sf.read(f)
                    if sr != self.sr:
                        x = librosa.resample(x, sr, self.sr)
                    self.noise.append(x)
                    self.n_noi += 1
                    self.noise_len = len(x)
        self.noise = np.array(self.noise)

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):

        data_idx = self.test_files[index]
        name = self.meta_data[data_idx][0]
        target = (int)(self.meta_data[data_idx][2])
        category = self.meta_data[data_idx][3]
        qpath = os.path.join(self.dir,name)
        query = self.load_segment(qpath)
        if self.noisy:
            noise = self.sample_noise()
            mixdown, query, noise = adjust_noise(noise, query, self.snr)
        else:
            query = rms_normalize(query)
            mixdown = query

        return (mixdown, query, target, category)

    def load_segment(self, path):
        full, sr = sf.read(path)
        pw = self.framelen - len(full)
        if len(full) < self.framelen:
            full = np.pad(full, pw, 'constant')
        rms = 0
        while rms < 1e-3:
            trim = np.random.randint(len(full) - self.framelen)
            seg = full[trim:trim+self.framelen]
            rms = cal_rms(seg)
        return seg

    def sample_noise(self):
        rs = np.random.randint(self.n_noi)
        rp = np.random.randint(self.noise_len - self.framelen)
        return self.noise[rs,rp:rp+self.framelen]

