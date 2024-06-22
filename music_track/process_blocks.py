from multiprocessing import Process, current_process
from threading import Thread
from queue import Queue
import time

import config as cfg
from logger import getmylogger
log = getmylogger(__name__)

class MusicDetector(Process):
    def __init__(self, *args, **kwargs):
        self.dtw = kwargs.pop('dtw_inst')
        self.live_queue = kwargs.pop('live_queue')
        self.md_event = kwargs.pop('md_event')
        self.mt_event = kwargs.pop('mt_event')
        super().__init__(*args, **kwargs)
        log.info(f"MusicDetector event status: {self.md_event.is_set(), self.mt_event.is_set()}")
        log.info(f"MusicDetector queue status: {self.live_queue.qsize()}")

    def run(self):
        p = current_process()
        log.info(f"New process -> {p.pid, p.name}")
        
        while self.md_event.is_set():
            # detecting
            try:
                live_feature = self.live_queue.get(timeout=60)
            except:
                # TODO: 如果過太久(ex. 60s)沒有計算到 直接進入tracking?
                log.warning(f"no live coming!")
                continue
            # live_feature = self.live_queue.get()
            detect_flag = self.dtw.run(live_feature)
            if detect_flag:
                self.md_event.clear()
                self.mt_event.set()
                
        log.info(f"mt_event: {self.mt_event.is_set()}")
        log.info(f"{p.pid, p.name} terminated")

class RoughEstimator(Process):
    def __init__(self, *args, **kwargs):
        # ref = kwargs.pop('ref')
        self.rpe = kwargs.pop('rpe_inst')
        self.live_queue = kwargs.pop('live_queue')
        self.possible_pos = kwargs.pop('possible_pos')
        self.acc_queue = kwargs.pop('acc_queue')
        self.mt_event = kwargs.pop('mt_event')

        super().__init__(*args, **kwargs)
    
    def run(self):
        p = current_process()
        log.info(f"New process -> {p.pid, p.name}")
        
        self.mt_event.wait()
        while self.mt_event.is_set():
            try:
                live_data = self.live_queue.get(timeout=1)
            except:
                continue
            low_feature, frame = live_data
            log.info(f"rough estimator start! {low_feature.shape, frame}")
            log.info(f"rough estimator frame {frame//cfg.LOW_HOP_SIZE+1}")
            
            # ref_point = self.acc_queue.peek_last()[1]//cfg.HOP_SIZE if len(self.acc_queue) > 0 else 0
            ref_point = None
            possible_frame = self.rpe.run(low_feature, ref_point)
            
            if possible_frame is not None:
                self.possible_pos.put((possible_frame[:8], frame))
                for pt in possible_frame[:8]:
                    self.rpe.rpe_point.append(((frame//cfg.LOW_HOP_SIZE)+1, (pt//cfg.LOW_HOP_SIZE)+1))
                # self.possible_pos.put((possible_frame, frame))
            log.info(f"rough estimator block end !")
            
        self.rpe.draw()
        log.info(f"{p.pid, p.name} terminated")

class DecisionMaker(Process):
    def __init__(self, *args, **kwargs):
        self.odtw = kwargs.pop('odtw_inst')
        self.rpe_queue = kwargs.pop('rpe_queue')
        self.acc_queue = kwargs.pop('acc_queue')
        self.mt_event = kwargs.pop('mt_event')
        super().__init__(*args, **kwargs)

    def run(self):
        p = current_process()
        log.info(f"New process -> {p.pid, p.name}")
        self.mt_event.wait()
        
        odtw_main_thread = Thread(target=self.odtw.run, args = (self.mt_event,))
        odtw_main_thread.start()
        
        jobs = Queue()
        while self.mt_event.is_set():
            try:
                rpe_list, rpe_live_frame = self.rpe_queue.get(timeout=2)
            except:
                # log.info(f"rpe res queue time out! Checking mt event!")
                continue
            
            log.info(f"rpe res data: {rpe_list, rpe_live_frame}") 
            for pos in rpe_list:
                jobs.put(int(pos))
                self.odtw.others_rpe_points.append((rpe_live_frame, pos))
            
            odtw_i, odtw_j = self.acc_queue.peek_last()
            log.info(f"now acc data: {odtw_i, odtw_j}") 
            
            # while odtw_i//cfg.HOP_SIZE < rpe_live_frame and odtw_main_thread.is_alive():
            #     odtw_i, odtw_j = self.acc_queue.peek_last()
            # prime_key_point = odtw_j//cfg.HOP_SIZE
            while odtw_i < rpe_live_frame and odtw_main_thread.is_alive():
                odtw_i, odtw_j = self.acc_queue.peek_last()
            prime_key_point = odtw_j
            jobs.put(prime_key_point) # now j_prime point
            # self.odtw.others_rpe_points.append((rpe_live_frame, prime_key_point))
            # for pos in range(prime_key_point-(cfg.MAX_RUN*5), min(prime_key_point+(cfg.MAX_RUN*5)+1, self.odtw.ref_len), 5):
            for pos in range(prime_key_point-(cfg.MAX_RUN*15), min(prime_key_point+(cfg.MAX_RUN*15)+1, self.odtw.ref_len), 15):
                jobs.put(pos)
                # self.odtw.others_rpe_points.append((rpe_live_frame, pos))
                self.odtw.others_back_rpe_points.append((rpe_live_frame, pos))
            workers = []
            for i in range(8):
                workers.append(Thread(target=self.odtw.deals_thread, args = (jobs, rpe_live_frame,)))
                # worker = threading.Thread(target=self.odtw.deals_thread, args = (jobs, rpe_live_frame,))
                workers[i].start()
            jobs.join()
            self.odtw.rpe_points.append((rpe_live_frame, self.odtw.min_rpe_ret[1]))
            log.info(f"workers end: {jobs.qsize()}")
            self.odtw.rpe_reset = True
        
        odtw_main_thread.join()
        log.info(f"odtw main thread end!")
        self.odtw.optimalWarpingPath()
        self.odtw.draw()
        log.info(f"{p.pid, p.name} terminated")