import numpy as np
import librosa
import time 

SAMPLE_RATE = 44100
WINDOW_SIZE = 2048
FRAME_SIZE = 84 # freq band

class FrameReader:
    def __init__(self):
        self.freq_map = self.makeFreqMap()
    
    def makeFreqMap(self):
        fmap = np.zeros(int(WINDOW_SIZE/2)+1)
        
        linear_band = 17
        fmap[:linear_band] = np.arange(0,linear_band)
        # 頻率下限
        f_min = 370  
        # 頻率上限 
        f_max = 12500
        # 分成的頻帶數
        n_bands = 66
        # 對數刻度
        f_scale = np.log10(f_max/f_min) 
        # 每個頻帶的頻率範圍比例
        f_ratio = np.power(10, f_scale / n_bands)

        log_fmap = np.array([f_min*np.power(f_ratio, i) for i in range(n_bands)])    
        band_width = SAMPLE_RATE/2.0/int(WINDOW_SIZE/2)
        for x in range(0, len(log_fmap)):
            lower_idx = int(log_fmap[x]/band_width)
            upper_idx = int(log_fmap[x+1]/band_width) if x != len(log_fmap)-1 else 580
            fmap[lower_idx:upper_idx] = 17+x
        fmap[580:] = 83

        # for f in fmap:
        #     print(f)
        # print(len(fmap))
        return fmap
    
    def readStream(self, s):
        new_y = np.zeros((FRAME_SIZE, s.shape[0]))
        for f in range(s.T.shape[0]): # freq 1025 -> 84
            new_y[int(self.freq_map[f])] += np.abs(s.T[f])
        new_y = new_y.T
        # for t in range(s.shape[0]):
        #     for f in range(s.shape[1]):# 1025 -> 84
        #         new_y[t][int(self.freq_map[f])] += np.abs(s[t][f])
        
        for t in range(1, new_y.shape[1]):
            differenceSum = np.sum(new_y[t])
            temp = (new_y[t]-new_y[t-1])>0.0
            new_y[t][temp==False] = 0.0
            
            rms = (sum(new_y[t]**2)/FRAME_SIZE)**0.5
            if rms < 0.0004:
                new_y[t][:] = 0.0
            if differenceSum > 0:
                new_y[t] = new_y[t]/differenceSum
                
        return new_y

# new_y = np.zeros((y.shape[0], FRAME_SIZE))
# print(new_y.shape)
# i=0
# while i != 20:
#     start = time.time()
#     print("start: ",start)
#     for t in range(y.shape[0]):
#         for f in range(y.shape[1]):
#             # print(new_y[t][int(framereader.freq_map[f])])
#             new_y[t][int(framereader.freq_map[f])] += np.abs(y[t][f])
#     print("end: ", time.time()-start)
#     i+=1
# print(new_y)
# print(new_y.shape)
