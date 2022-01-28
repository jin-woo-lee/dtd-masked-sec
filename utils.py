import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

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

def minmax_normalize(x):
    b, c, f, t = x.shape
    x_min = x.reshape(b,c*f*t).min(dim=-1).values.reshape(b,1,1,1)
    x = x - x_min
    x_max = x.reshape(b,c*f*t).max(dim=-1).values.reshape(b,1,1,1)
    x = x / x_max
    return x

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def adjust_noise(noise, source, snr, divide_by_max = True):
    eps = np.finfo(np.float32).eps
    noise_rms = cal_rms(noise) # noise rms

    num = cal_rms(source) # source rms
    den = np.power(10., snr/20)
    desired_noise_rms = num/den

    # calculate gain
    try:
        gain = desired_noise_rms / (noise_rms + eps)
    except OverflowError:
        gain = 1.

    noise = gain*noise

    mix = source + noise

    if divide_by_max == True:
        mix_max_val = np.abs(mix).max(axis=-1)
        src_max_val = np.abs(source).max(axis=-1)
        noise_max_val = np.abs(noise).max(axis=-1)

        if (mix_max_val > 1.) or (src_max_val > 1.) or (noise_max_val > 1.):
            max_val = np.max([mix_max_val, src_max_val, noise_max_val])
            mix = mix / (max_val+eps)
            source = source / (max_val+eps)
            noise = noise / (max_val+eps)
        else:
            pass
    else:
        pass

    return mix, source, noise

def rms_normalize(wav, ref_dB=-23.0):
    # RMS normalize
    eps = np.finfo(np.float32).eps
    rms = cal_rms(wav)
    rms_dB = 20*np.log(rms/1) # rms_dB
    ref_linear = np.power(10, ref_dB/20.)
    gain = ref_linear / np.sqrt(np.mean(np.square(wav), axis=-1) + eps)
    wav = gain * wav
    return wav

