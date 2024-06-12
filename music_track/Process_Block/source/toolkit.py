import librosa
from source import config as cfg
import numpy as np

def dict_inf():
    # return (float('inf'),)
    return np.inf

def timeFreqStft(y):
    y = librosa.stft(y, n_fft=cfg.STFT['nfft'], hop_length=cfg.STFT['hop'], win_length=cfg.STFT['window'])
    return y.T

def extractHighAndLowFeature(audio):
    high_feature = librosa.feature.melspectrogram(y=audio, sr=cfg.SAMPLE_RATE, n_fft=cfg.WINDOW_SIZE, hop_length=cfg.HOP_SIZE, n_mels=cfg.FRAME_SIZE, fmax=8000, center=False)
    
    low_window_size = 30 # 600ms (0.6*44100)/(0.02*44100)
    low_hop_size = 15 # 300ms (0.3*44100)/(0.02*44100)
    new_hanning = np.hanning(low_window_size)
    low_feature = np.array([np.convolve(high_feature[i], new_hanning, mode='same')[::low_hop_size] for i in range(high_feature.shape[0])], dtype=np.float32)
    
    high_feature = np.diff(high_feature, axis=1) # 對time做差值 取聲音變化的特徵
    # high_feature = librosa.power_to_db(np.abs(high_feature)**2, ref=np.max)
    
    low_feature = np.diff(low_feature, axis=1) # 對time做差值 取聲音變化的特徵
    # low_feature = librosa.power_to_db(np.abs(low_feature)**2, ref=np.max)
    
    return high_feature.T, low_feature.T # t-f
    # return high_feature.T, None# t-f