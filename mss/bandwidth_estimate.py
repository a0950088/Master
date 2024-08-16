import os
import librosa
import numpy as np

dataset_path = './bandwidth_dataset/violin' # 頻帶切割資料集路徑

filelist = os.listdir(dataset_path)
print(filelist)
freq_band = np.zeros(22050)

'''
迭帶單一樂器的所有音訊資料計算fft頻率成份分布`freq_band`
'''
for file in filelist:
    print("processing: ", file)
    data, sr = librosa.load(dataset_path+'/'+file, sr=44100)
    samples_p = len(data)
    fft = np.fft.fft(data)
    amplitudes = 2 / samples_p * np.abs(fft)
    frequencies = np.fft.fftfreq(samples_p, d=1/sr)
    for sample in range(samples_p//2):
        freq_band[round(frequencies[sample])-1]+=amplitudes[sample]

total = np.sum(freq_band) # 加總所有頻率成份的值
max_b = np.max(freq_band) # 找出最多成份的頻率所擁有的值
VIOLIN_CUT_PROPORTION = (max_b/total)*14 # 設定小提琴的切割比例閥值(手動調整閥值)
PIANO_CUT_PROPORTION = (max_b/total)*1.05 # 設定鋼琴的切割比例閥值(手動調整閥值)
MIN_FREQ_CUT = 100 # 最小頻率切割值

cut_point = 0
freq_val = 0
total_freq_val = 0
prev_i = 0
for i in range(0, 22050):
    # 使用 VIOLIN_CUT_PROPORTION 和 PIANO_CUT_PROPORTION 計算兩種樂器的切割點
    if freq_val >= VIOLIN_CUT_PROPORTION and (i-prev_i) >= MIN_FREQ_CUT:
        cut_point+=1
        freq_val = 0
        prev_i = i
        print(f"cut_point {i}: ", cut_point)
    
    val = np.sum(freq_band[i])/total
    freq_val+=val
    total_freq_val+=val


# 畫fft頻譜圖
# violin_path='./bandwidth_dataset/2243.wav'
# piano_path='./bandwidth_dataset/2529.wav'

# violin, sr = librosa.load(violin_path, sr=44100)
# piano, _ = librosa.load(piano_path, sr=44100)
# samples_v = len(violin)

# time sec
# v_t0, v_t1 = 0, samples_v/sr
# p_t0, p_t1 = 0, samples_p/sr
# v_xs = np.linspace(v_t0, v_t1, samples_v)
# p_xs = np.linspace(p_t0, p_t1, samples_p)
# ax1.plot(v_xs, violin)
# ax2.plot(p_xs, piano)
# violin_fft = np.fft.fft(violin)
# violin_amplitudes = 2 / samples_v * np.abs(violin_fft)
# # violin_frequencies = np.fft.fftfreq(samples_v) * samples_v * 1 / (v_t1 - v_t0)
# violin_frequencies = np.fft.fftfreq(samples_v, d=1/sr)
# # ax3.semilogx(violin_frequencies[:len(violin_frequencies) // 2], violin_amplitudes[:len(violin_fft) // 2])
# for sample in range(samples_v//2):
#     violin_freq_band[round(violin_frequencies[sample])-1]+=violin_amplitudes[sample]

# ax3.bar([i for i in range(22050)], [j for j in violin_freq_band])
# print(freq_band)

# piano_fft = np.fft.fft(piano)
# piano_amplitudes = 2 / samples_p * np.abs(piano_fft)
# piano_frequencies = np.fft.fftfreq(samples_p, d=1/sr)
# print(np.sum(piano_amplitudes))
# for sample in range(samples_p//2):
#     piano_freq_band[round(piano_frequencies[sample])-1]+=piano_amplitudes[sample]

# print(piano_freq_band.shape)
# print(np.sum(piano_freq_band))
# total = np.sum(piano_freq_band)
# cut_point = 0
# temp = 0
# temp2 = 0
# for i in range(0, 22050, 100):
#     # print(np.sum(piano_freq_band[i:i+50]))
#     if temp >= 0.01:
#         cut_point+=1
#         temp = 0
#         print(f"cut_point {i}: ", cut_point)
    
#     val = np.sum(piano_freq_band[i:i+100])/total
#     temp+=val
#     temp2+=val
        
#     print(f"total {i}: ",val, temp, temp2)