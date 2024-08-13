import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
import torchaudio
import pretty_midi
from museval.metrics import bss_eval
import soundfile as sf

def get_stft_res(y, sr, frame=2048, hop_size=512, win_len=2048):
    y = librosa.stft(y, n_fft=frame, hop_length=hop_size, win_length=win_len)
    y = np.abs(y)**2 # 絕對值平方
    ylog = librosa.amplitude_to_db(y)
    return ylog

def get_istft_res(y, sr, frame=2048, hop_size=512, win_len=2048):
    y_hat = librosa.istft(y, n_fft=frame, hop_length=hop_size, win_length=win_len)
    return y_hat

def get_fft_res():
    
    violin_path='./bandwidth_dataset/violin/2241.wav'
    piano_path='./bandwidth_dataset/piano/2372.wav'
    
    violin, sr = librosa.load(violin_path, sr=44100)
    piano, _ = librosa.load(piano_path, sr=44100)
    samples_v = len(violin)
    samples_p = len(piano)
    violin_freq_band = np.zeros(22050)
    piano_freq_band = np.zeros(22050)
    
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(221)
    ax1.set_title('(a) violin_solo_waveform', y=-0.2)
    ax2 = fig.add_subplot(222)
    ax2.set_title('(b) piano_solo_waveform', y=-0.2)
    ax3 = fig.add_subplot(223)
    ax3.set_title('(c) violin_solo_fft', y=-0.2)
    ax4 = fig.add_subplot(224)
    ax4.set_title('(d) piano_solo_fft', y=-0.2)
    
    # time sec
    v_t0, v_t1 = 0, samples_v/sr
    p_t0, p_t1 = 0, samples_p/sr
    v_xs = np.linspace(v_t0, v_t1, samples_v)
    p_xs = np.linspace(p_t0, p_t1, samples_p)
    ax1.plot(v_xs, violin)
    ax2.plot(p_xs, piano)
    violin_fft = np.fft.fft(violin)
    violin_amplitudes = 2 / samples_v * np.abs(violin_fft)
    # violin_frequencies = np.fft.fftfreq(samples_v) * samples_v * 1 / (v_t1 - v_t0)
    violin_frequencies = np.fft.fftfreq(samples_v, d=1/sr)
    # ax3.semilogx(violin_frequencies[:len(violin_frequencies) // 2], violin_amplitudes[:len(violin_fft) // 2])
    for sample in range(samples_v//2):
        violin_freq_band[round(violin_frequencies[sample])-1]+=violin_amplitudes[sample]
    
    print(np.max(violin_freq_band))
    # ax3.set_ylim([0, 0.05])
    ax3.bar([i for i in range(22050)], [j for j in violin_freq_band])
    # print(freq_band)
    
    piano_fft = np.fft.fft(piano)
    piano_amplitudes = 2 / samples_p * np.abs(piano_fft)
    piano_frequencies = np.fft.fftfreq(samples_p, d=1/sr)
    # print(np.sum(piano_amplitudes))
    for sample in range(samples_p//2):
        piano_freq_band[round(piano_frequencies[sample])-1]+=piano_amplitudes[sample]
    
    ax4.bar([i for i in range(22050)], [j for j in piano_freq_band])

    plt.show()
get_fft_res()