import librosa
import numpy as np
import config as cfg
import collections
from multiprocessing.managers import BaseManager
from pydub import AudioSegment

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

def extractHighAndLowFeature(audio):
    high_feature = librosa.feature.melspectrogram(y=audio, sr=cfg.SAMPLE_RATE, n_fft=cfg.WINDOW_SIZE, hop_length=cfg.HOP_SIZE, n_mels=cfg.FRAME_SIZE, fmax=8000)
    
    low_window_size = 30 # 600ms (0.6*44100)/(0.02*44100)
    low_hop_size = 15 # 300ms (0.3*44100)/(0.02*44100)
    new_hanning = np.hanning(low_window_size)
    low_feature = np.array([np.convolve(high_feature[i], new_hanning, mode='same')[::low_hop_size] for i in range(high_feature.shape[0])], dtype=np.float32)
    
    high_feature = np.diff(high_feature, axis=1) # 對time做差值 取聲音變化的特徵
    high_feature = librosa.power_to_db(np.abs(high_feature)**2, ref=np.max)
    
    low_feature = np.diff(low_feature, axis=1) # 對time做差值 取聲音變化的特徵
    low_feature = librosa.power_to_db(np.abs(low_feature)**2, ref=np.max)
    
    return high_feature.T, low_feature.T # t-f

class DequeManager(BaseManager):
    pass

class DequeProxy():
    def __init__(self, *args, **kwargs):
        maxlen = kwargs.pop('maxlen')
        self.deque = collections.deque(maxlen=maxlen)
        
    def __len__(self):
        return self.deque.__len__()
    
    def appendleft(self, x):
        self.deque.appendleft(x)
        
    def append(self, x):
        self.deque.append(x)
        
    def pop(self):
        return self.deque.pop()
    
    def popleft(self):
        return self.deque.popleft()

    def peek_last(self):
        return self.deque[-1]
    
def overlay(path1, path2, out_path, adjust_volume=(None,None)):
    sound1 = AudioSegment.from_file(path1)
    sound2 = AudioSegment.from_file(path2)
    if adjust_volume[0] is not None:
        if adjust_volume[0] == 1:
            sound1 = sound1[:]+adjust_volume[1]
        else:
            sound2 = sound2[:]+adjust_volume[1]
    combined = sound1.overlay(sound2)
    combined.export(out_path, format='wav')