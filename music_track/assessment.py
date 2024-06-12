import madmom
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile
import mido
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from pathlib import Path

# 思考dtw的時間對應: 
# 開頭結尾都是一樣的，所以理論上 (1,1),(2,2) ... (n,n) 會是最好的對齊
# est_idx > live_idx -> live 比 est 還要快
# est_idx < live_idx -> live 比 est 還要慢

# TODO: Music detector

SAMPLE_RATE = 44100
WINDOW_SIZE = int(0.046*SAMPLE_RATE)
HOP_SIZE = int(0.020*SAMPLE_RATE) # 20ms

def draw_res(time_frame, latency, time_slice, time_frame_bpm, bpm, avg_deviation):
    plt.figure(figsize=(15, 9))
    host = host_subplot(111, axes_class=axisartist.Axes)
    
    par1 = host.twinx()

    par1.axis["right"].toggle(all=True)

    p1, = host.plot(time_frame, latency, label="latency")
    p2, = par1.plot(time_frame, time_frame_bpm, label="bpm")

    host.set_xlim(0, len(time_frame))
    host.set_xticks(time_slice, [f"m{bar}" for bar in range(1, len(time_slice)+1)])
    # host.set_ylim(min(latency), max(latency))
    host.set_ylim(-7, 7)
    par1.set_ylim(min(bpm), max(bpm))

    host.set_xlabel("number of bar")
    host.set_ylabel("number of Sixteenth note")
    par1.set_ylabel("bpm")

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    host.axis["bottom"].label.set_fontsize(15)
    host.axis["left"].label.set_fontsize(15)
    par1.axis["right"].label.set_fontsize(15)
    
    plt.grid()
    plt.title(f"Avarage Deviation: {avg_deviation} ms", fontsize='xx-large',fontweight='heavy')
    plt.plot()
    plt.savefig(f"{folder}/latency.png")
    plt.show()

def midoTempoAndBpmConvert(num):
    return 6*1e7/num

def getEstMidiTimeInfo(main_track):
    tick_count = 0
    shot = 480 # tick
    one_bar = 4*shot
    continued_sec = 0
    now_tempo_sec = 0
    # bin_count = 1
    
    bar_slice = [0]
    bpm_change_list = []
    for msg in main_track:
        # print("msg: ", vars(msg), isinstance(msg, mido.MetaMessage))
        if msg.type == 'set_tempo':
            now_tempo_sec = msg.tempo*1e-6
            bpm_change_list.append(((midoTempoAndBpmConvert(msg.tempo)), round((continued_sec*SAMPLE_RATE)/HOP_SIZE)))
        if msg.type == 'note_on':
            tick_count += msg.time
            continued_sec += msg.time*now_tempo_sec/shot # note持續時間/一拍時間 = ?拍 -> 轉換成秒
            # print("tick_count:", tick_count)
            if tick_count-one_bar == 0: # 累積一小節
                # print("one_bar !!!", continued_sec)
                bar_slice.append(round((continued_sec*SAMPLE_RATE)/HOP_SIZE)) # 小節結束時間(sec)
                # bin_count+=1
                tick_count = 0
    # 補上最後一小節結束時間
    add_offset = one_bar-tick_count
    continued_sec += add_offset*now_tempo_sec/shot
    bar_slice.append(round((continued_sec*SAMPLE_RATE)/HOP_SIZE))
    print("bar count: ", len(bar_slice))
    print("bar slice: ", bar_slice)
    print("midi continued sec: ", continued_sec)
    
    return bar_slice, bpm_change_list

midi_path = Path('./assessment/slow/data/live_slow.mid')
live_path = Path('./assessment/slow/tracking_result/live_slow/2024-06-03_ref/acc_record.wav')
est_path = Path('./assessment/slow/data/est_slow.wav')
folder = Path(f"./{midi_path.parent.parent}/assessment_result/{midi_path.stem}")
folder.mkdir(parents=True, exist_ok=True)
# midi_path = './assessment/normal/live_normal_v2.mid'
# live_path = './assessment/normal/LiveAcc_live_normal_v2.wav'
# est_path = './assessment/normal/est_normal_v2.wav'
# midi_path = './assessment/slow/live_slow_v2.mid'
# live_path = './assessment/slow/LiveAcc_live_slow_v2.wav'
# est_path = './assessment/slow/est_slow_v2.wav'
# midi_path = './assessment/fast/live_fast.mid'
# live_path = './assessment/fast/LiveAcc_live_fast.wav'
# est_path = './assessment/fast/est_fast.wav'

mid = mido.MidiFile(str(midi_path), clip=True)
violin_track = mid.tracks[0]

# compute bin position
bar_slice, bpm_change_list = getEstMidiTimeInfo(violin_track)

live, _ = librosa.core.load(str(live_path), sr=44100)
est, _ = librosa.core.load(str(est_path), sr=44100)
cut_offset = round((bar_slice[-1]+1)*HOP_SIZE)
live = live[:cut_offset]
est = est[:cut_offset]
print(live.shape, est.shape)

# test madmom
# proc = madmom.features.tempo.TempoEstimationProcessor(min_bpm=80, max_bpm=160, fps=100)
# act = madmom.features.beats.RNNBeatProcessor()(est_path)
# beats_p = proc(act)
# print(beats_p)
# print(beats_p.shape)
# for i in range(len(beats_p)):
#     if i % 100-1 == 0 and i != 0:
#         print(plus)
#         plus = 0
#     plus+=beats_p[i]
# print(len(beats_p))

# dtw feature
x_1_chroma = librosa.feature.chroma_stft(y=live, sr=SAMPLE_RATE, tuning=0, norm=2,
                                         hop_length=HOP_SIZE, n_fft=WINDOW_SIZE)
x_2_chroma = librosa.feature.chroma_stft(y=est, sr=SAMPLE_RATE, tuning=0, norm=2,
                                         hop_length=HOP_SIZE, n_fft=WINDOW_SIZE)
print("chroma feature shape: ", x_1_chroma.shape, x_2_chroma.shape)

# D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='euclidean')
print("dtw shape:", D.shape, wp.shape)
# prev_n = -1
# for n,m in wp:
#     if prev_n != n:
#         print(n, m-n)
#     else:
#         print("same!")
#     prev_n = n
# prev_m = -1
# for n,m in wp:
#     print(n,m)
    # if prev_m != m:
    #     print(n, m, m-n)
    # else:
    #     print("same!")
    # prev_m = m

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# # librosa.display.specshow(D, x_axis='frames', y_axis='frames',
# #                         hop_length=HOP_SIZE, n_fft=WINDOW_SIZE, sr=SAMPLE_RATE)
# # imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
# #                  origin='lower', interpolation='nearest', aspect='auto')
# plt.imshow(D, cmap="inferno")
# plt.plot(wp[:, 0], wp[:, 1], marker='o', color='g', markersize = 1)
# plt.title('Warping Path on Acc. Cost Matrix $D$')
# plt.colorbar()
# plt.xlabel('live')
# plt.ylabel('ref')
# plt.gca().invert_yaxis()

# plt.show()

prev_est = -1
host_x = []
host_y = []
for live_idx, est_idx in wp:
    if prev_est == est_idx:
        # print("continue")
        continue
    # print(est_idx, live_idx, est_idx-live_idx)
    prev_est = est_idx
    host_x.insert(0, est_idx)
    # host_y.insert(0, ((live_idx-est_idx)*HOP_SIZE)/SAMPLE_RATE)
    # host_y.insert(0, (live_idx-est_idx))
    # host_y.insert(0, (live_idx-est_idx))
    host_y.insert(0, (est_idx-live_idx))
    # live_idx < est_idx: faster than est
    # live_idx > est_idx: slower than est

# compute average deviation
total_frame = 0
for frame in host_y[:bar_slice[-1]]:
    total_frame+=np.abs(frame)
avg_deviation = (((total_frame/bar_slice[-1])*HOP_SIZE)/SAMPLE_RATE)*1e+3 # ms

# 消除連續重複的bpm值 ex. (80,0)(81,0)(81,74) -> (80,0)(81,74) 畫圖用
new_list = []
for bpm, frame in bpm_change_list:
    if new_list == []:
        new_list.append([bpm, frame])
    elif new_list[-1][0] == bpm:
        new_list[-1][1] = frame
    else:
        new_list.append([bpm, frame])

# 填滿time_frame，畫圖用
bpm_x = []
bpm_y = []
prev_bpm = new_list[0][0]
prev_frame = new_list[0][1]
if prev_frame != 0:
    temp = [prev_bpm]*(prev_frame)
    bpm_x+=temp
bpm_y.append(new_list[0][0])
for bpm, frame in new_list[1:]:
    bpm_y.append(bpm)
    temp = [prev_bpm]*(frame-prev_frame)
    bpm_x += temp
    prev_bpm = bpm
    prev_frame = frame
temp = [bpm_y[-1]]*(x_1_chroma.shape[1]-len(bpm_x))
bpm_x+=temp
# print(len(bpm_x))
# print(bpm_x)

# 計算每個frame的latency，用16分音符作為單位 畫圖用
idx = 0
while idx < len(host_y):
    note16_sec = midoTempoAndBpmConvert(bpm_x[idx])*1e-6/4
    host_y[idx] = ((host_y[idx]*HOP_SIZE)/SAMPLE_RATE)/note16_sec
    idx+=1
# print(min(host_y), max(host_y))
print(f"avg_deviation: {avg_deviation} ms")
draw_res(host_x, host_y, bar_slice, bpm_x, bpm_y, avg_deviation)