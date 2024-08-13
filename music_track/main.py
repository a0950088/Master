import time
import librosa
import numpy as np

import config as cfg
from stream import Stream
from methods import DTW, RPE, ODTW
from data_manager import ProcQueue, ProcEvent, AudioData
from process_blocks import MusicDetector, RoughEstimator, DecisionMaker

from logger import getmylogger
log = getmylogger(__name__)

if __name__ == '__main__':
    '''
    mode = test 使用預先錄好的音檔進行追蹤測試
    mode = live 追蹤現場演奏
    '''
    log.info(f"Mode: {cfg.MODE}")
    if cfg.MODE == 'test':
        live, _ = librosa.load(str(cfg.LIVE_PATH), sr=44100) # return amplitude
        ref, _ = librosa.load(str(cfg.REF_PATH), sr=44100)
        acc, _ = librosa.load(str(cfg.ACC_PATH), sr=44100)
    else:
        live = None
        ref, _ = librosa.load(str(cfg.REF_PATH), sr=44100)
        acc, _ = librosa.load(str(cfg.ACC_PATH), sr=44100)
    
    '''初始化Process用到的event和queue'''
    proc_event = ProcEvent()
    proc_queue = ProcQueue()
    
    md_event = proc_event.getMDEvent()
    mt_event = proc_event.getMTEvent()
    
    live_queue = proc_queue.getLiveQueue() # [livedata, len(livedata)]
    acc_queue = proc_queue.getAccDeque()
    output_queue = proc_queue.getOutputDeque()
    
    '''初始化所有音訊資料'''
    audio_data = AudioData(ref, acc, 
                           proc_queue.getMDQueue(),
                           proc_queue.getRPEFeatureQueue(),
                           proc_queue.getODTWFeatureQueue(),
                           proc_queue.getResPathQueue(),
                           md_event,
                           mt_event)
    
    '''初始化串流系統'''
    audio_stream = Stream(live_queue=live_queue, output_queue=output_queue, test_data=live)
    
    '''初始化三個Process和方法'''
    dtw_inst = DTW(audio_data.getMDHighRefFeature(), audio_data.getMDSilenceFeature())
    md_proc = MusicDetector(dtw_inst=dtw_inst, live_queue=proc_queue.getMDQueue(), 
                            md_event=md_event, mt_event=mt_event)
    
    rpe_inst = RPE(audio_data.getMTLOWRefFeature()) # use normal feature
    rpe_proc = RoughEstimator(rpe_inst=rpe_inst, live_queue=proc_queue.getRPEFeatureQueue(), acc_queue=proc_queue.getAccDeque(),
                              possible_pos=proc_queue.getRPEResQueue(), mt_event=mt_event)
    
    odtw_inst = ODTW(audio_data.getMTHighRefFeature(), proc_queue.getODTWFeatureQueue(), 
                          proc_queue.getAccDeque(), res_path_queue=proc_queue.getResPathQueue()) # use normal feature
    dma_proc = DecisionMaker(odtw_inst=odtw_inst,
                            rpe_queue=proc_queue.getRPEResQueue(), acc_queue=proc_queue.getAccDeque(),
                            mt_event=mt_event)
    
    '''啟動Processes'''
    md_proc.start()
    rpe_proc.start()
    dma_proc.start()
    time.sleep(5) # 等待5秒，確保所有Process都有啟動
    log.info(f"Active processes complete")
    
    live_len = 0 # 紀錄現場音訊總長度
    window_point = 0 # 高解析度特徵切割點
    low_window_point = 0 # 低解析度特徵切割點
    record = np.zeros(0, dtype=np.float32) # 紀錄現場音訊
    live_end_count = cfg.LIVE_END_COUNT # 認定現場音訊結束的常數
    
    '''啟動即時串流'''
    audio_stream.start()
    while audio_stream.stream.is_active():
        prev_run_time = time.time() # 紀錄每個live frame計算的時間
        try:
            live_frame = live_queue.get(timeout=2)
        except:
            log.warning(f"no live coming!")
            break
        live_data = live_frame[0]
        live_len += live_frame[1]
        
        '''如果music tracking event啟動，開始判斷現場音訊是否結束'''
        if mt_event.is_set():
            log.warning(f"Testing mute seg: {audio_data.detectMuteLiveSegment(live_data)}, {live_len}")
            if audio_data.detectMuteLiveSegment(live_data):
                live_end_count -= 1
                log.warning(f"live end count: {live_end_count}")
                if live_end_count <= 0:
                    # end live
                    log.warning(f"END LIVE")
                    output_queue.append(None)
            else:
                live_end_count = cfg.LIVE_END_COUNT
                
        record = np.concatenate((record, live_data), dtype=np.float32)
        
        '''偵測現場音訊是否開始'''
        if md_event.is_set() and len(record) >= cfg.HALF_SEC_FRAME:
            audio_data.onlineMdFeatureExtraction(record[-cfg.HALF_SEC_FRAME:])
        
        '''計算高低解析度特徵'''
        while window_point*cfg.HOP_SIZE+cfg.WINDOW_SIZE < len(record):
            segment = record[window_point*cfg.HOP_SIZE:window_point*cfg.HOP_SIZE+cfg.WINDOW_SIZE]
            # queue add audio_data.live_high_feature[window_point]
            audio_data.onlineHighFeatureExtraction(segment)
            if audio_data.onlineLowFeatureExtraction(low_window_point, window_point):
                # queue add audio_data.live_low_feature[(low_window_point-1)*2:low_window_point*2]
                low_window_point+=1
            window_point+=1
        
        '''輸出伴奏'''
        if len(acc_queue) > 0:
            _, acc_pos = acc_queue.peek_last()
            output_data = audio_data.getOutputSegment(acc_pos)
            output_queue.append(output_data)
        audio_data.logAudioQueueSize()
        log.info(f"now tracking window point: {window_point-audio_data.mt_start_frame}")
        log.info(f"Live queue status: {live_queue.qsize(), live_len}")
        log.info(f"Spent time: {time.time()-prev_run_time}")

    '''結束每個process'''
    md_proc.join()
    rpe_proc.join()
    dma_proc.join()
    
    '''輸出線上追蹤路徑'''
    audio_data.getOutputPath()
    
    '''輸出音訊特徵圖、離線與線上特徵比較圖'''
    audio_data.draw_feature(diff=False)
    audio_data.drawOfflineAndOnlineOutputFrames()
    
    '''輸出現場音訊與伴奏音訊混合後的結果音訊'''
    # audio_data.writeOutputAudio(record, live)
    audio_data.writeOutputAudio(record, audio_stream.live_record, audio_stream.output_record)
    
    log.info(f"All proc end")