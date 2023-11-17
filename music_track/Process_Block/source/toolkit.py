import librosa
import config as cfg

def dict_inf():
    return (float('inf'),)

def timeFreqStft(y):
    y = librosa.stft(y, n_fft=cfg.STFT['nfft'], hop_length=cfg.STFT['hop'], win_length=cfg.STFT['window'])
    return y.T