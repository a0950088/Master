from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
import numpy as np
import time
from source import config as cfg
from source.rpe import RPE, RPELOW

class RoughEstimator(Process):
    def __init__(self, *args, **kwargs):
        ref = kwargs.pop('ref')
        self.live_queue = kwargs.pop('live_queue')
        self.possible_pos = kwargs.pop('possible_pos')
        self.acc_queue = kwargs.pop('acc_queue')
        self.status = kwargs.pop('status')
        # self.share_data = kwargs.pop('share_data')
        # self.rpe = RPE(ref)
        self.rpe = RPELOW(ref)
        # self.rpe.run(run_flag=False)
        super().__init__(*args, **kwargs)
    
    def run(self):
        p = current_process()
        print("New process -> [%s] %s" % (p.pid, p.name))
        live_data = []
        while True:
            # if self.live_queue.empty():
            #     continue
            start = time.time()
            live_data = self.live_queue.get()
            if live_data is None:
                break
            live, frame = live_data
            print("rough estimator start! ", live.shape, frame)
            ref_point = self.acc_queue.peek_last()[1]//cfg.HOP_SIZE if len(self.acc_queue) > 0 else 0
            possible_frame = self.rpe.run(ref_point, live_frame=live)
                
            if possible_frame is not None:
                # self.possible_pos.append((possible_frame[:4], frame))
                self.possible_pos.append((possible_frame[:8], frame))
                # self.possible_pos.append((possible_frame, frame))
            print("possible_frame ", possible_frame)
            # print("possible_frame: ", len(possible_frame))
            # print("possible_queue len: ", self.possible_pos.qsize())
            print("rough estimator block end: ", time.time()-start)
        print("[%s] %s terminated" % (p.pid, p.name))