from multiprocessing import Manager, Event
from multiprocessing.managers import BaseManager
from collections import deque
from librosa import stft, power_to_db, griffinlim, istft
from librosa.feature import melspectrogram
from librosa.sequence import dtw
from librosa.display import specshow
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from pydub import AudioSegment

import config as cfg
from logger import getmylogger
log = getmylogger(__name__)

class AudioData():
    def __init__(self, ref, acc,
                 md_q, rpe_feature_q, odtw_feature_q, res_path_q, 
                 md_event, mt_event):
        self.ref = ref
        self.acc = acc
        
        self.music_detector_queue = md_q
        self.low_feature_queue = rpe_feature_q
        self.high_feature_queue = odtw_feature_q
        self.res_path_queue = res_path_q
        
        self.music_detector_event = md_event
        self.music_trackers_event = mt_event
        
        self.live_stft_feature = np.zeros(((cfg.WINDOW_SIZE//2)+1, 0), dtype=np.float32)
        self.live_high_feature = np.zeros((cfg.FRAME_SIZE, 0), dtype=np.float32)
        self.live_high_diff_feature = np.zeros((cfg.FRAME_SIZE, 0), dtype=np.float32)
        self.live_low_feature = np.zeros((cfg.FRAME_SIZE, 0), dtype=np.float32)
        self.live_low_diff_feature = np.zeros((cfg.FRAME_SIZE, 0), dtype=np.float32)
        
        self.ref_stft_feature = self.getSTFTFeature(self.ref)
        self.ref_high_feature = self.getHighFeature(self.ref_stft_feature)
        self.ref_low_feature = self.getLowFeature(self.ref_high_feature)
        
        self.acc_stft_feature = self.getSTFTFeature(self.acc)
        
        silence_feature = self.getSTFTFeature(np.zeros(cfg.HALF_SEC_FRAME, dtype=np.float32))
        self.silence_feature = self.getHighFeature(silence_feature)
        # self.md_ref_feature = self.ref_high_feature[:, :self.silence_feature.shape[1]]
        self.half_sec_ref_mean_amplitude = np.abs(self.ref[:cfg.HALF_SEC_FRAME]).mean()
        self.half_sec_ref_RMS = np.sqrt(np.mean(self.ref[:cfg.HALF_SEC_FRAME] ** 2))
        
        self.mt_start_frame = 0
        self.mt_adjust_amplitude = 1
        
        self.output_path = []
        
        log.info(f"MD mean amplitude/RMS: {self.half_sec_ref_mean_amplitude, self.half_sec_ref_RMS}")
        log.info(f"Ref data shape: {self.ref_stft_feature.shape, self.ref_high_feature.shape, self.ref_low_feature.shape}")
        log.info(f"Acc data shape: {self.acc_stft_feature.shape}")
        log.info(f"Event status: {self.music_detector_event.is_set(), self.music_trackers_event.is_set()}")
        log.info(f"Queue status: {self.music_detector_queue.qsize(), self.low_feature_queue.qsize(), self.high_feature_queue.qsize()}")
    
    def logAudioQueueSize(self):
        log.info(f"audio queue size: {self.music_detector_queue.qsize(), self.high_feature_queue.qsize(), self.low_feature_queue.qsize()}")
    
    def onlineMdFeatureExtraction(self, half_seg):
        seg_mean_amplitude = np.abs(half_seg).mean()
        seg_rms = np.sqrt(np.mean(half_seg ** 2))
        log.info(f"seg_mean_amplitude: {seg_mean_amplitude}, seg RMS: {seg_rms}")
        # if seg_mean_amplitude <= cfg.MEAN_AMPLITUDE_THRESHOLD:
        #     return
        if seg_rms <= cfg.RMS_THRESHOLD or seg_mean_amplitude <= cfg.MEAN_AMPLITUDE_THRESHOLD:
            return
        
        # self.mt_adjust_amplitude = self.half_sec_ref_mean_amplitude/seg_mean_amplitude
        # if self.mt_adjust_amplitude <= cfg.MIN_ADJUST_MAG:
        #     self.mt_adjust_amplitude = cfg.MIN_ADJUST_MAG
        # elif self.mt_adjust_amplitude >= cfg.MAX_ADJUST_MAG:
        #     self.mt_adjust_amplitude = cfg.MAX_ADJUST_MAG
        
        '''先不要調整amplitude'''
        # log.info(f"mt_adjust_amplitude: {self.mt_adjust_amplitude}")
        # half_seg *= self.mt_adjust_amplitude
        
        stft_feature = self.getSTFTFeature(half_seg)
        high_feature = self.getHighFeature(stft_feature)
        
        self.music_detector_queue.put(high_feature)
        # log.info(f"music_detector_queue: {self.music_detector_queue.qsize()}")
    
    def onlineHighFeatureExtraction(self, seg):
        if len(seg) < cfg.WINDOW_SIZE:
            seg = np.concatenate((seg, np.zeros(cfg.WINDOW_SIZE-len(seg), dtype=np.float32)), dtype=np.float32)
        if self.music_trackers_event.is_set():
            seg *= self.mt_adjust_amplitude
        
        stft_feature = self.getSTFTFeature(seg)
        high_feature = self.getHighFeature(stft_feature)
        self.live_stft_feature = np.concatenate((self.live_stft_feature, stft_feature), axis=1)
        self.live_high_feature = np.concatenate((self.live_high_feature, high_feature), axis=1)
        
        if self.music_trackers_event.is_set():
            # use normal feature
            self.high_feature_queue.put(high_feature.T)
        else:
            self.mt_start_frame = self.live_high_feature.shape[1]
            log.info(f"mt_start_frame: {self.mt_start_frame}")
    
    def onlineLowFeatureExtraction(self, low_window_point, window_point):
        start = low_window_point*cfg.LOW_HOP_SIZE
        end = start+cfg.LOW_WINDOWS_SIZE
        if self.live_high_feature.shape[1]%end == 0:
            low_seg = self.live_high_feature[:, start:end]
            low_feature = np.array([np.convolve(low_seg[i],
                                                cfg.LOW_WIN, 
                                                mode='same') for i in range(cfg.FRAME_SIZE)], 
                                   dtype=np.float32)
            
            low_feature = np.sum(low_feature, axis=1) / cfg.LOW_WINDOWS_SIZE
            low_feature = low_feature[:, np.newaxis]
            
            self.live_low_feature = np.concatenate((self.live_low_feature, low_feature), axis=1)
            
            if self.music_trackers_event.is_set():
                # use normal feature
                self.low_feature_queue.put((low_feature.T, window_point-self.mt_start_frame)) # odtw was computed by 0 frame
            return True
        return False
        
        
    def getSTFTFeature(self, raw_frame_data):
        return stft(y=raw_frame_data, 
                    n_fft=cfg.NFFT, 
                    hop_length=cfg.HOP_SIZE, 
                    win_length=cfg.WINDOW_SIZE, 
                    center=False)
    
    def getHighFeature(self, stft_feature):
        return melspectrogram(S=np.abs(stft_feature)**2, 
                              n_fft=cfg.NFFT, 
                              hop_length=cfg.HOP_SIZE, 
                              n_mels=cfg.FRAME_SIZE, 
                              fmax=8000)
    
    def getLowFeature(self, high_feature):
        low_feature = np.zeros((cfg.FRAME_SIZE, 0), dtype=np.float32)
        for t in range(0, high_feature.shape[1], cfg.LOW_HOP_SIZE):
            new = np.array([np.convolve(high_feature[i][t:t+cfg.LOW_WINDOWS_SIZE], cfg.LOW_WIN, mode='same') for i in range(cfg.FRAME_SIZE)], dtype=np.float32)
            new = np.sum(new, axis=1) / cfg.LOW_WINDOWS_SIZE

            low_feature = np.concatenate((low_feature, new[:, np.newaxis]), axis=1)
        return low_feature

    def getMDHighRefFeature(self):
        return self.ref_high_feature[:, :self.silence_feature.shape[1]]

    def getMDSilenceFeature(self):
        return self.silence_feature
    
    def getMTHighRefFeature(self):
        return self.ref_high_feature[:, cfg.HALF_SEC_HIGH_FEATURE:].T
    
    def getMTLOWRefFeature(self):
        return self.ref_low_feature[:, cfg.HALF_SEC_LOW_FEATURE:].T
    
    def getOutputPath(self):
        for i in range(self.mt_start_frame):
            self.output_path.append(np.array([i, 0]))
        while not self.res_path_queue.empty():
            i, j = self.res_path_queue.get()
            self.output_path.append(np.array([i+self.mt_start_frame, j]))
        self.output_path = np.array(self.output_path)
        
    def drawOfflineAndOnlineOutputFrames(self):
        D, wp = dtw(X=self.live_high_feature, Y=self.ref_high_feature, metric='euclidean')
        count = 0
        avg_deviation = 0
        prei = -1
        # print(wp, type(wp), len(wp))
        new_offline_path = []
        for i,j in wp:
            if i == prei:
                continue
            print(i)
            avg_deviation += abs(self.output_path[i][1]-j)
            prei = i
            count += 1
            new_offline_path.append((i,j))
        avg_deviation /= count
        print(count)
        print(len(self.output_path))
        print("avg_deviation: ", avg_deviation)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.imshow(D.T, cmap="inferno")
        for i in range(len(self.output_path)):
            print(self.output_path[i], new_offline_path[len(self.output_path)-1-i])
            
        plt.plot(self.output_path[:, 0], self.output_path[:, 1], marker='o', color='green', markersize = 1)
        plt.plot(wp[:, 0], wp[:, 1], marker='o', color='lightskyblue', markersize = 1)
        # plt.scatter(wp[:, 0], wp[:, 1], marker='o', c='lightskyblue')
        # plt.scatter(self.output_path[:, 0], self.output_path[:, 1], marker='o', c='green')
        plt.title(f"Offline(blue) & Online(green) Path\n Avarage Deviation Frames: {avg_deviation} frames\n Avarage Deviation ms: {avg_deviation*cfg.HOP_SIZE/cfg.SAMPLE_RATE*1000} ms\n")
        plt.colorbar()
        plt.xlabel('live frame')
        plt.ylabel('output frame')
        plt.gca().invert_yaxis()
        plt.savefig(f"{cfg.FOLDER}/offline_and_online_output.png")
        # plt.show()
    
    def draw_feature(self, diff=False):
        live_high = self.live_high_feature
        live_low = self.live_low_feature
        ref_high = self.ref_high_feature
        ref_low = self.ref_low_feature
            
        live_high = power_to_db(np.abs(live_high)**2, ref=np.max)
        live_low = power_to_db(np.abs(live_low)**2, ref=np.max)
        fig = plt.figure(figsize=(24, 18))
        ax1 = fig.add_subplot(221)
        ax1.set_title('live_high_feature')
        ax2 = fig.add_subplot(222)
        ax2.set_title('live_low_feature')
        img_l = specshow(live_high, sr=cfg.SAMPLE_RATE, hop_length=cfg.HOP_SIZE, x_axis='frames', y_axis='mel', fmax=8000, ax=ax1)
        img2_l = specshow(live_low, sr=cfg.SAMPLE_RATE, hop_length=13230, x_axis='frames', y_axis='mel', fmax=8000, ax=ax2)
        # plt.colorbar(img_l, ax=[ax1,ax2], format='%+2.f')
        # plt.colorbar(img, ax=[ax1,], format='%+2.f')
        # plt.savefig(f"{folder}/live_feature.png")
        # plt.show()
        
        # _, ref_high, ref_low = offlineFeatureExtraction(ref)
        # print(ref_high.shape, ref_low.shape)
        ref_high = power_to_db(np.abs(ref_high)**2, ref=np.max)
        ref_low = power_to_db(np.abs(ref_low)**2, ref=np.max)
        # fig = plt.figure(figsize=(15, 9))
        ax3 = fig.add_subplot(223)
        ax3.set_title('ref_high_feature')
        ax4 = fig.add_subplot(224)
        ax4.set_title('ref_low_feature')
        img_r = specshow(ref_high, sr=cfg.SAMPLE_RATE, hop_length=cfg.HOP_SIZE, x_axis='frames', y_axis='mel', fmax=8000, ax=ax3)
        img2_r = specshow(ref_low, sr=cfg.SAMPLE_RATE, hop_length=13230, x_axis='frames', y_axis='mel', fmax=8000, ax=ax4)
        plt.colorbar(img_l, ax=[ax1,ax2,ax3,ax4], format='%+2.f')
        # plt.savefig(f"{folder}/ref_feature.png")
        plt.savefig(f"{cfg.FOLDER}/feature.png")
        # plt.show()
    
    
    def writeOutputAudio(self, record, live):
        adjust_acc_frames = np.zeros((self.mt_start_frame, (cfg.WINDOW_SIZE//2)+1), dtype=np.float32)
    
        original_acc_frames = self.acc_stft_feature[:, cfg.HALF_SEC_HIGH_FEATURE:].T
        for i, j in self.output_path[self.mt_start_frame:]:
            if j < original_acc_frames.shape[0]:
                adjust_acc_frames = np.concatenate((adjust_acc_frames, original_acc_frames[np.newaxis,j]))
        # y_inv = griffinlim(adjust_acc_frames.T, hop_length=cfg.HOP_SIZE, win_length=cfg.WINDOW_SIZE, n_fft=cfg.NFFT, center=False)
        adjust_acc_audio = istft(adjust_acc_frames.T, n_fft=cfg.NFFT, hop_length=cfg.HOP_SIZE, win_length=cfg.WINDOW_SIZE, center=False)
        
        # while not self.res_path_queue.empty():
        #     i,j = self.res_path_queue.get()
        #     if j < original_acc_frames.shape[0]:
        #         adjust_acc_frames = np.concatenate((adjust_acc_frames, original_acc_frames[np.newaxis,j]))
        # adjust_acc_audio = istft(adjust_acc_frames.T, n_fft=cfg.NFFT, hop_length=cfg.HOP_SIZE, win_length=cfg.WINDOW_SIZE, center=False)
        
        # ''' # write result
        output_main_record = f"{cfg.FOLDER}/output_main_record.wav"
        output_live = f"{cfg.FOLDER}/live_record.wav"
        output_liveacc = f"{cfg.FOLDER}/acc_record.wav"
        # scipy.io.wavfile.write(output_live, 44100, np.array(stream_live_audio, dtype=np.float32))
        scipy.io.wavfile.write(output_main_record, 44100, np.array(record, dtype=np.float32))
        scipy.io.wavfile.write(output_live, 44100, np.array(live, dtype=np.float32))
        scipy.io.wavfile.write(output_liveacc, 44100, np.array(adjust_acc_audio, dtype=np.float32))
        output_combined = f"{cfg.FOLDER}/combined.wav"
        
        # overlay live and acc audio
        sound1 = AudioSegment.from_file(output_live)
        sound2 = AudioSegment.from_file(output_liveacc)
        sound2 = sound2[:]+5
        combined = sound1.overlay(sound2)
        combined.export(output_combined, format='wav')
    
class ProcEvent():
    def __init__(self):
        self.music_detector_event = Event() # initial True
        self.music_trackers_event = Event() # initial False
        
        self.music_detector_event.set()
    
    def getMDEvent(self):
        return self.music_detector_event
    
    def getMTEvent(self):
        return self.music_trackers_event
        
class ProcQueue():
    def __init__(self):
        self.__manager = Manager()
        
        self.live_q = self.__manager.Queue()
        self.md_q = self.__manager.Queue()
        self.odtw_feature_q = self.__manager.Queue()
        self.rpe_feature_q = self.__manager.Queue()
        self.rpe_res_q = self.__manager.Queue()
        self.res_path_q = self.__manager.Queue()
        self.acc_record_q = self.__manager.Queue()
        
        # active share data deque manager
        DequeManager.register('DequeProxy', DequeProxy, exposed=['__len__', 'append', 'appendleft',
                                                            'pop', 'popleft', 'peek_last'])
        deque_manager = DequeManager()
        deque_manager.start()
        self.acc_deque = deque_manager.DequeProxy(maxlen=3)
        
    def getLiveQueue(self):
        return self.live_q

    def getMDQueue(self):
        return self.md_q

    def getODTWFeatureQueue(self):
        return self.odtw_feature_q

    def getRPEFeatureQueue(self):
        return self.rpe_feature_q

    def getRPEResQueue(self):
        return self.rpe_res_q

    def getResPathQueue(self):
        return self.res_path_q

    def getAccRecordQueue(self):
        return self.acc_record_q

    def getAccDeque(self):
        return self.acc_deque

class DequeManager(BaseManager):
    pass

class DequeProxy():
    def __init__(self, *args, **kwargs):
        maxlen = kwargs.pop('maxlen')
        self.deque = deque(maxlen=maxlen)
        
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