import librosa
import numpy as np
import config as cfg

def timeFreqStft(y):
    y = librosa.stft(y, n_fft=cfg.STFT['nfft'], hop_length=cfg.STFT['hop'], win_length=cfg.STFT['window'])
    return y.T

def getLiveDataBySec(data, sec):
    need_sec = int(cfg.ONE_BUFFER_SEC*sec)
    start_idx = len(data) - need_sec
    if start_idx >= 0:
        # newest n sec live data
        return data[len(data)-need_sec:]
    else:
        # concatenate silent data in front of live data
        return np.concatenate((np.zeros(abs(start_idx)), data), dtype=np.float32)
    
# def liveAndRefDist(live, ref, i):
    # return (sum(abs(live-ref)**2)**0.5, ref, i)
def liveAndRefDist(live, ref, i, end_i):
    return (sum(abs(live-ref)**2)**0.5, i, end_i)