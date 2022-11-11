import os
from os import walk
from random import sample

from torch.utils.data import Dataset
import torch

import numpy as np
from scipy import signal
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import read as readwav
import librosa

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg, FigureCanvas

import pickle as pk

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['xtick.labelsize'] = 23
matplotlib.rcParams['ytick.labelsize'] = 23

# Constants
n_fft = 512
hop_length = 256
SR = 16000
over_sample = 4
res_factor = 0.8
octaves = 6
notes_per_octave=10
cdict  = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'alpha':  ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
        }

my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
plt.register_cmap(cmap=my_mask)


class NSynth_data(torch.utils.data.Dataset):
    def __init__(self, dir, files):
        self.data = np.array([librosa.core.load(dir + f, sr=None)[0]
                              for f in files])
        self.files = files

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        amplitude = self.data[idx].reshape(1,-1)
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude)
        return amplitude


class MusicalAudio(torch.utils.data.Dataset):
    def __init__(self, dir, files, n_samples=64000, sr=16000):
        self.n_samples = n_samples
        self.files = files
        self.sr = sr
        self.trimmed_sorted = []
        data = np.zeros((1, n_samples))

        for file in files:
            wav, _ = librosa.core.load(dir + file, sr=sr)
            samples = self.trim_file(wav)
            self.trimmed_sorted.append(samples)
            data = np.vstack((data, samples))
        self.data = data[1:]

    def trim_file(self, wav_data):
        ts = len(wav_data)
        split_indices = range(self.n_samples, ts, self.n_samples)
        wav_split = np.split(wav_data, split_indices)

        # Pad last sample
        if ts % self.n_samples != 0:
            diff = self.n_samples - len(wav_split[-1])
            wav_split[-1] = np.pad(wav_split[-1], (0,diff))

        data = np.vstack(wav_split)

        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        amplitude = self.data[idx].reshape(1,-1)
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude)
        return amplitude


def get_random_set(path, n, batch_size=32, shuffle=False,
                    num_workers=0, cat='nsynth', names=[]):
    """
    Returns a dataloader and the names of the samples.
    """
    if not names:
        for (_, _, filenames) in walk(path):
            names.extend(filenames)
            break

    if n == None:
        n = len(names)
    samples = sample(names, n)

    if cat == 'nsynth':
        data = NSynth_data(path, samples)
    else:
        data = MusicalAudio(path, samples)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                    num_workers=num_workers, shuffle=shuffle)

    # Split samples names like batch (from GeeksforGeeks).
    samples = [samples[i:i + batch_size]
               for i in range(0, len(samples), batch_size)]

    return train_loader, np.array(samples, dtype=object)


def fig_to_im(fig, canvas):
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width),3)
    plt.close(fig)

    return image


def plot_waves(wavor, wavdec, save_path=None):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8),
                           sharex=True, constrained_layout=True)
    canvas = FigureCanvas(fig)

    wavor = wavor.squeeze()
    wavdec = wavdec.squeeze()

    # Plot audio waves.
    ax[0].set_title(f"Original audio waveform ({16000}Hz)", fontsize=26)
    ax[0].plot(np.linspace(0, len(wavor)/16000, len(wavor)), wavor)
    ax[1].set_title("Spectrogram of original audio", fontsize=26)
    ax[1].specgram(wavor, Fs=16000, NFFT=n_fft, cmap='magma')

    image1 = fig_to_im(fig, canvas)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8),
                           sharex=True, constrained_layout=True)
    canvas = FigureCanvas(fig)

    # Plot spectrogram of audio signals.
    ax[0].set_title(f"Reconstructed audio waveform ({16000}Hz)", fontsize=26)
    ax[0].plot(np.linspace(0, len(wavdec)/16000, len(wavdec)), wavdec)
    
    ax[1].set_title("Spectrogram of decoded audio", fontsize=26)
    ax[1].specgram(wavdec, Fs=16000, NFFT=n_fft, cmap='magma')

    if save_path:
        fig.savefig(save_path + "plots", format='png')

    image2 = fig_to_im(fig, canvas)

    return image1, image2


def save_waves(wavor, wavdec, path):
    wavfile.write(path + 'original', 16000, wavor)
    wavfile.write(path + 'decoded', 16000, wavdec)

    return None


def create_mask(batch_shape, n):
    max_idx = batch_shape[-1]

    idx = torch.LongTensor()
    for i in range(batch_shape[0]):
        idx = torch.cat((idx, torch.randperm(max_idx)[:n].unsqueeze(0)), dim=0)

    return idx


def create_log(batch, data, model, device, names, batch_index=-1):
    if isinstance(data.dataset, NSynth_data):
        N = 5
        subset = sample(list(range(len(batch))), N)

        with torch.no_grad():
            original = batch[subset].to(device)
            decoded = model(original, None, device)

        original = original.cpu().numpy().squeeze()
        decoded = decoded.cpu().numpy().squeeze()
        freq_spect = plot_waves(original.reshape(-1,1), decoded.reshape(-1,1))

        all_samples = np.concatenate([original, decoded]).reshape(int(2*N), -1)

        rows = 2
        cols = N
        row_labels = ['Original', 'Reconstructed']
        col_labels = np.array(names[batch_index], dtype=object)[subset]
        print(col_labels)
        col_labels = [s.split('_')[0] + '\n' + s.split('_')[1] for s in col_labels]
        sep_waves = {}
        for label, ori, dec in zip(col_labels, original, decoded):
            sep_waves[label] = plot_waves(ori, dec)
    else:
        original = sample(data.dataset.trimmed_sorted, 1)[0]
        for i, _ in enumerate(original):
            original[i] = original[i] / np.max(abs(original[i]))

        inp = torch.from_numpy(original).unsqueeze(1).to(device)
        with torch.no_grad():
            decoded = model(inp, None, device).cpu().numpy()

        rows = len(original)
        row_labels = []
        cols = 2
        col_labels = ['Original', 'Reconstruction', 'difference']

        empty_lists = [[] for _ in range(cols)]
        zipped = zip(original, decoded, empty_lists)
        zip_flat = []
        for z in zipped:
            zip_flat.extend(z)

        original = np.concatenate(original)
        decoded = np.concatenate(decoded.squeeze())
        all_samples = []
        for o, d in zip(original, decoded):
            all_samples.append(o)
            all_samples.append(d)

        freq_spect = plot_waves(original.reshape(-1,), decoded.reshape(-1,))

    rainbowgrams = plot_notes(all_samples, rows=rows, cols=cols,
                              row_labels=row_labels, col_labels=col_labels)

    log = {
        'original': original,
        'decoded': decoded,
        'all_samples': all_samples,
        'rows': rows,
        'cols': cols,
        'row_labels': row_labels,
        'col_labels': col_labels,
        'freq_spect': freq_spect,
        'rainbowgrams': rainbowgrams,
        'separated_waves': sep_waves
    }

    plt.close()
    return log

#---------------------------------------------------------------------
# Functions below are retrieved from:
# https://gist.github.com/jesseengel/e223622e255bd5b8c9130407397a0494
#---------------------------------------------------------------------

# Plotting functions


def note_specgram(path, ax, peak=70.0, use_cqt=True):
    sr = 0
    # Add several samples together
    if isinstance(path, list):
        for i, p in enumerate(path):
            sr, a = readwav(p)
            audio = a if i == 0 else a + audio
    elif isinstance(path, np.ndarray):
        audio, sr = path, SR
    # Load one sample
    else:    
        sr, audio = readwav(path)


    # audio = audio.astype(np.float32)
    if use_cqt:
        C = librosa.cqt(audio, sr=sr, hop_length=hop_length, 
                        bins_per_octave=int(notes_per_octave*over_sample), 
                        n_bins=int(octaves * notes_per_octave * over_sample), 
                        filter_scale=res_factor, 
                        fmin=librosa.note_to_hz('C2'))
    else:
        C = librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=hop_length,
                        center=True)

    mag, phase = librosa.core.magphase(C)
    phase_angle = np.angle(phase)
    phase_unwrapped = np.unwrap(phase_angle)
    dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
    mag = (librosa.power_to_db(mag**2, amin=1e-13, top_db=peak,
                            ref=np.max) / peak) + 1

    ax.imshow(dphase[::-1, :], cmap=plt.cm.rainbow)
    ax.imshow(mag[::-1, :], cmap=my_mask)


def plot_notes(list_of_paths, rows=2, cols=4, col_labels=[], row_labels=[],
              use_cqt=True, peak=70.0):
    """Build a CQT rowsXcols.
    """
    column = 0
    N = len(list_of_paths)
    assert N == rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(18,8), sharex=True, sharey=True)
    # fig.subplots_adjust(left=0.1, right=0.9, wspace=0.1, hspace=0.1)

    # fig = plt.figure(figsize=(18, N * 1.25))
    for i, path in enumerate(list_of_paths):
        row = int(i / cols)
        col = i % cols
        if rows == 1:
            ax = [axes][col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        note_specgram(path, ax, peak, use_cqt)

        ax.set_facecolor('white')
        ax.set_xticks([]); ax.set_yticks([])
        
        if col == 0 and row_labels:
            ax.set_ylabel(row_labels[row], fontsize=26)
        if row == rows-1 and col_labels:
            ax.set_xlabel(col_labels[col], fontsize=26)

    canvas = FigureCanvas(fig)
    image = fig_to_im(fig, canvas)

    return image