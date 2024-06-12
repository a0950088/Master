from source.toolkit import extractHighAndLowFeature
from stream import TestStream
from util import DequeManager, DequeProxy, overlay
from rough_estimator import RoughEstimator
from decision_maker import DMA
from pathlib import Path

import librosa
import time
import numpy as np
from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
# from multiprocessing.managers import BaseManager
# import threading
import matplotlib.pyplot as plt
import scipy.io.wavfile
import collections

if __name__ == "__main__":
    # test
    print("test")
    # live, _ = librosa.load('../music/test/summer3rd_violin_15s.wav', sr=44100)
    # ref, _ = librosa.load('../music/test/summer3rd_violin_15s_1m08s.wav', sr=44100)
    # live, _ = librosa.load('../Canon_bpm72_13s.wav', sr=44100)
    # ref, _ = librosa.load('../Canon_bpm62_15s.wav', sr=44100)
    # acc, _ = librosa.load('../Canon_piano_bpm62_15s.wav', sr=44100)
    # load audio
    
    # live_path = '../Canon_bpm62_15s.wav'
    # ref_path = '../Canon_bpm72_13s.wav'
    # acc_path = '../Canon_piano_bpm72_13s.wav'
    live_path = '../assessment/live_bpm110_12bins_clear.wav'
    ref_path = '../assessment/ref_12bins_clear.wav'
    acc_path = '../assessment/acc_12bins_clear.wav'
    # live_path = '../real_record/Canon/bpm62/Canon_bpm62_novibrato_bow_mainpart-1.wav'
    # ref_path = '../real_record/Canon/bpm72/Canon_bpm72_vibrato_splitbow_mainpart-1.wav'
    # acc_path = '../real_record/Canon/bpm72/Canon_bpm72_acc-1.wav'
    # live_path = '../real_record/Canon/bpm72/Canon_bpm72_novibrato_onepos-1.wav'
    # ref_path = '../real_record/Canon/bpm62/Canon_bpm62_novibrato_onepos-1.wav'
    # acc_path = '../real_record/Canon/bpm62/Canon_bpm62_acc-1.wav'

    live, _ = librosa.load(live_path, sr=44100)
    ref, _ = librosa.load(ref_path, sr=44100)
    acc, _ = librosa.load(acc_path, sr=44100)
    # live = np.concatenate((live, np.zeros(2048, dtype=np.float32)), dtype=np.float32)
    # ref = np.concatenate((ref, np.zeros(2048, dtype=np.float32)), dtype=np.float32)
    # acc = np.concatenate((acc, np.zeros(2048, dtype=np.float32)), dtype=np.float32)
    print(live.shape, ref.shape)
    high_live, _ = extractHighAndLowFeature(live)
    high_ref, _ = extractHighAndLowFeature(ref)
    print(high_live.shape, high_ref.shape)
    
    # active share data manager
    manager = Manager()
    live_queue = manager.Queue()
    odtw_live_queue = manager.Queue()
    rpe_live_queue = manager.Queue()
    rpe_queue = manager.Queue()
    acc_record_queue = manager.Queue()

    proc_status = manager.dict({
        'rpe_proc': False,
        'dma_proc': False
    })
    
    acc_record = np.zeros(0, dtype=np.float32)
    share_acc_record = manager.dict({
        'acc': acc,
        'acc_record': acc_record
    })
    
    # active share data deque manager
    DequeManager.register('DequeProxy', DequeProxy, exposed=['__len__', 'append', 'appendleft',
                                                            'pop', 'popleft', 'peek_last'])
    deque_manager = DequeManager()
    deque_manager.start()
    acc_queue = deque_manager.DequeProxy(maxlen=3)
    test_rpe_queue = deque_manager.DequeProxy(maxlen=3)

    # active processes
    testst = TestStream(live, acc, live_queue, acc_queue, share_acc_record)
    rpe_proc = RoughEstimator(ref=ref, live_queue=rpe_live_queue, possible_pos=test_rpe_queue, acc_queue=acc_queue, status=proc_status)
    dma_proc = DMA(ref=ref, acc=acc, live_queue=odtw_live_queue, rpe_queue=test_rpe_queue, acc_queue=acc_queue, share_acc_record=share_acc_record, status=proc_status)
    print("proc ready")
    
    rpe_proc.start()
    dma_proc.start()
    time.sleep(5)
    
    # start tracking
    record = np.zeros(0, dtype=np.float32)
    live_len = 0
    prev_live_len = 0
    print("proc start")
    testst.start()
    odtw_f = 0
    while testst.stream.is_active():
        # print("main live len: ", live_len, live_len-prev_live_len, live_queue.qsize(), rpe_live_queue.qsize(), odtw_live_queue.qsize(), len(acc_queue))
        try:
            live_frame = live_queue.get(timeout=3)
        except:
            print("no live coming!")
            live_frame = None
        if live_frame is None:
            rpe_live_queue.put(None)
            odtw_live_queue.put(None)
            break
        live_data = live_frame[0]
        live_len += live_frame[1]
        record = np.concatenate((record, live_data), dtype=np.float32)
        if prev_live_len+17640 <= live_len: # 17640->0.4s
            # put to queue
            high_feature, _ = extractHighAndLowFeature(record[prev_live_len:])
            odtw_f+=high_feature.shape[0]
            odtw_live_queue.put((high_feature, live_len))
            rpe_live_queue.put((record[prev_live_len:], live_len))
            prev_live_len = live_len
    print("main proc end")
    print(odtw_f)
    rpe_proc.join()
    dma_proc.join()
    print("proc join")
    
    # acc_record = np.zeros(0, dtype=np.float32)
    # while not acc_record_queue.empty():
    #     pos = acc_record_queue.get()
    #     if pos is None:
    #         acc_record = np.concatenate((acc_record, np.zeros(882, dtype=np.float32)), dtype=np.float32)
    #     else:
    #         acc_data = acc[pos:pos+cfg.HOP_SIZE]
    #         acc_record = np.concatenate((acc_record, acc_data), dtype=np.float32)
    
    # folder = Path(f"../odtw_test/{Path(live_path).stem}_{Path(ref_path).stem}")
    # folder.mkdir(parents=True, exist_ok=True)
    # output_live = f"{folder}/Live_{Path(live_path).stem}.wav"
    # output_liveacc = f"{folder}/Liveacc_{Path(acc_path).stem}.wav"
    # # scipy.io.wavfile.write("acc_record.wav", 44100, np.array(share_acc_record['acc_record'], dtype=np.float32))
    
    # scipy.io.wavfile.write(output_live, 44100, np.array(record, dtype=np.float32))
    # # scipy.io.wavfile.write(output_liveacc, 44100, np.array(testst.live_acc, dtype=np.float32))
    # scipy.io.wavfile.write(output_liveacc, 44100, np.array(testst.live_acc[18432:], dtype=np.float32))
    # # output_combined = f"{folder}/combined.wav"
    # output_combined = f"{folder}/combined_remove_delay18432.wav"
    # overlay(output_live, output_liveacc, output_combined)
    
    # print(testst.path)