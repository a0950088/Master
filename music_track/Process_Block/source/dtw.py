import numpy as np
import librosa
import time
from collections import defaultdict
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io.wavfile
from toolkit import dict_inf

class DTW:
    def __init__(self):
        self.radius = 1
        self.min_time_size = self.radius+2 # odd better
        self.speed_factor = 1
        self.normalized = 0.0
        
    def _compute_dist(self, x, y):
        return np.sum(np.abs(x-y)**2)**0.5
    
    def _reduce_by_half(self, x):
        ret = []
        for i in range(0, len(x), 2):
            if i+1 >= len(x):
                ret.append(x[i])
            else:
                ret.append((x[i]+x[i+1])/2)
        return ret
    
    def _expand_window(self, path, lenx, leny):
        path_ = set(path)
        for i, j in path:
            for a, b in ((i + a, j + b)
                        for a in range(-self.radius, self.radius+1)
                        for b in range(-self.radius, self.radius+1)):
                path_.add((a, b))
                
        _window = set()
        for i, j in path_:
            for a, b in ((i*2, j*2), (i*2, j*2 + 1),
                        (i*2+1, j*2), (i*2+1, j*2+1)):
                _window.add((a, b))

        window = []
        start_j = 0
        for i in range(0, lenx):
            new_start_j = None
            for j in range(start_j, leny):
                if (i, j) in _window:
                    window.append((i, j))
                    if new_start_j is None:
                        new_start_j = j
                elif new_start_j is not None:
                    break
            start_j = new_start_j
        return window
    
    def _execute(self, x, y, window=None):
        lenx, leny = len(x), len(y)
        if not window:
            window = [(i,j) for i in range(lenx) for j in range(leny)] # (0,0) ~ (lenx-1,leny-1)
        window = ((i+1,j+1) for i,j in window)
        
        d_matrix = defaultdict(dict_inf)
        d_matrix[0,0] = (0,0,0) # dist, x_idx, y_idx
        for i,j in window:
            dt = self._compute_dist(x[i-1], y[j-1])
            d_matrix[i, j] = min((d_matrix[i-1, j][0]+dt, i-1, j),
                                 (d_matrix[i, j-1][0]+dt, i, j-1),
                                 (d_matrix[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])

        path = []
        i, j = lenx, leny 
        while not (i == j == 0):
            path.append((i-1, j-1))
            i, j = d_matrix[i, j][1], d_matrix[i, j][2]
        path.reverse()
        # return (d_matrix[lenx, leny][0]/(self.t1len*self.t2len), path)
        return (d_matrix[lenx, leny][0], path)
        # return (d_matrix[lenx, leny][0]/self.normalized, path)
        
    def fastdtw(self, x, y):
        if len(x)<self.min_time_size or len(y)<self.min_time_size:
            return self._execute(x, y)
        
        half_x = self._reduce_by_half(x)
        half_y = self._reduce_by_half(y)
        dist, path = self.fastdtw(half_x, half_y)
        window = self._expand_window(path, len(x), len(y))
        return self._execute(x, y, window)
        
    def dtw(self, x, y):
        return self._execute(x, y)
    
    def run(self, type, t1, t2):
        # print(type, t1, t2)
        self._t1, self._t2 = t1, t2
        self.t1len = self._t1.shape[0]
        self.t2len = self._t2.shape[0]
        # Calculate distance matrix
        # Trackback from end to start
        if type == "DTW":
            rd, path = self.dtw(self._t1, self._t2)
        else:
            rd, path = self.fastdtw(self._t1, self._t2)
        return rd, path
    
def get_stft_res(y, frame=2048, hop_size=512, win_len=2048):
    y = librosa.stft(y, n_fft=frame, hop_length=hop_size, win_length=win_len)
    return y

def get_istft_res(y, frame=2048, hop_size=512, win_len=2048):
    y_hat = librosa.istft(y, n_fft=frame, hop_length=hop_size, win_length=win_len)
    return y_hat

def get_filtered_res(fixed_t, est_t, ref_t, path=None, est_mindb=1, ref_maxdb=0.3):
    pre_x = -1
    count = 0
    value = 0
    for x,y in path:
        if x == pre_x or pre_x == -1:
            value += ref_t[y]
            count+=1
        else:
            value = ref_t[y]
            count=0
        est = est_t[x] >= est_mindb
        ref = value/count < ref_maxdb if count != 0 else value < ref_maxdb
        bool_filter = est & ref
        fixed_t[x][bool_filter==True] = 0.0
        pre_x = x
    fixed_y = fixed_t.T

    return fixed_y

def show_compare_spec(est_spec, ref_spec, fixed_spec=None):
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(131)
    ax1.set_title('est_spec')
    ax2 = fig.add_subplot(132)
    ax2.set_title('ref_spec')
    img1 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(est_spec)**2, ref=np.max), hop_length=512, y_axis='log', x_axis='frames', ax=ax1, cmap="hot", n_fft=2048, win_length=2048) #cmap="hot"
    img2 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(ref_spec)**2, ref=np.max), hop_length=512, y_axis='log', x_axis='frames', ax=ax2, cmap="hot", n_fft=2048, win_length=2048) #cmap="hot"
    ax3 = fig.add_subplot(133)
    ax3.set_title('fixed_spec')
    img3 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(fixed_spec)**2, ref=np.max), hop_length=512, y_axis='log', x_axis='frames', ax=ax3, cmap="hot", n_fft=2048, win_length=2048) #cmap="hot"
    plt.colorbar(img3, ax=[ax1,ax2,ax3], format='%+2.f')
    
    # if fixed_spec != None:
    #     ax3 = fig.add_subplot(133)
    #     ax3.set_title('fixed_spec')
    #     img3 = librosa.display.specshow(np.abs(fixed_spec), hop_length=hop_size, y_axis='log', x_axis='frames', ax=ax3, cmap="hot", n_fft=frame, win_length=win_length) #cmap="hot"
    #     plt.colorbar(img3, ax=[ax1,ax2,ax3], format='%+2.f')
    # else:
    #     plt.colorbar(img2, ax=[ax1,ax2], format='%+2.f')

    plt.show()

def stretch(x, factor, nfft=2048):
    # x: PCM float32 format, return np.ndarray
    stft = librosa.core.stft(x, n_fft=nfft).transpose()
    stft_rows = stft.shape[0] # time
    stft_cols = stft.shape[1] # freq.

    times = np.arange(0, stft.shape[0], factor)
    hop = nfft/4
    phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ nfft
    stft = np.concatenate((stft, np.zeros((1, stft_cols))), axis=0)

    indices = np.floor(times).astype(np.int16)
    alpha = np.expand_dims(times - np.floor(times), axis=1)
    mag = (1. - alpha) * np.absolute(stft[indices, :]) + alpha * np.absolute(stft[indices + 1, :])
    dphi = np.angle(stft[indices + 1, :]) - np.angle(stft[indices, :]) - phase_adv
    dphi = dphi - 2 * np.pi * np.floor(dphi/(2 * np.pi))
    phase_adv_acc = np.matmul(np.expand_dims(np.arange(len(times) + 1),axis=1), np.expand_dims(phase_adv, axis=0))
    phase = np.concatenate( (np.zeros((1, stft_cols)), np.cumsum(dphi, axis=0)), axis=0) + phase_adv_acc
    phase += np.angle(stft[0, :])
    stft_new = mag * np.exp(phase[:-1,:]*1j)
    return librosa.core.istft(stft_new.transpose())

# seriesA = 'test_live.wav'
# seriesB = 'summer3rd_violin_15s.wav'
# sa, sr = librosa.load(seriesA, sr=44100)
# sb, _ = librosa.load(seriesB, sr=44100)

# sa_st = get_stft_res(sa)
# sb_st = get_stft_res(sb)
# fixed_spec_t = sa_st.T.copy()
# print(sa_st.shape)
# print(sb_st.shape)
# sai = np.abs(sa_st).T
# sbi = np.abs(sb_st).T
# dtw = DTW()
# rd, path = dtw.run("fDTW",abs(sa_st).T, abs(sb_st).T)
# # rd, path = fastdtw(sai,sbi)
# # print(sai.shape)
# print(rd)
# print(len(path))
# pre_i = None
# pre_j = None
# factor = 1
# for i,j in path:
#     if pre_i != i and pre_j != j:
#         # good alignment
#         pre_i = i
#         pre_j = j
#         continue
#     elif pre_i == i and pre_j != j:
#         # live is faster
#         factor-=0.001
#     elif pre_i != i and pre_j == j: 
#         # live is slower
#         factor+=0.001
#     if i == len(abs(sa_st).T)-1 or j == len(abs(sb_st).T)-1:
#         break
#     pre_i = i
#     pre_j = j
#     # print(pre_i,pre_j)
    
# print(factor)
# y=stretch(sa, factor)
# scipy.io.wavfile.write("stretch.wav", 44100, np.array(y, dtype=np.float32))
#     if i == len(abs(sa_st).T)-1:
#         break
#     else:
# fixed_spec = get_filtered_res(fixed_spec_t, sai, sbi, path)
# fixed_data = get_istft_res(fixed_spec)
# sf.write('new_dtw.wav', fixed_data, sr)
# show_compare_spec(sa_st, sb_st, fixed_spec)
# print(path)