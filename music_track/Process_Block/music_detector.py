from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
import numpy as np
import time
from .source import DTW
from .util import timeFreqStft, getLiveDataBySec
import config as cfg

class MusicDetector(Process):
    def __init__(self, *args, **kwargs):
        self.live_queue = kwargs.pop('queue')
        self.share_data = kwargs.pop('share_data')
        init_ref = self.share_data['ref_data']
        self.ref_data = timeFreqStft(init_ref[:int(cfg.ONE_BUFFER_SEC*0.5)])
        self.dtw = DTW()
        self.ret = []
        self.acc_start = False
        self.threshold = 10.0
        super().__init__(*args, **kwargs)

    def run(self):
        p = current_process()
        print("New process -> [%s] %s" % (p.pid, p.name))
        print("music_detector")
        while not self.acc_start:
            # normal
            start = time.time()
            live_data = self.live_queue.get()
            print("music_detector start! ", live_data.shape)
            live = timeFreqStft(getLiveDataBySec(live_data, sec=0.5))
            rd, _ = self.dtw.run("DTW", live, self.ref_data)
            print("cost: ", rd)
            self.ret.append(rd)
            print("music detector end: ", time.time()-start)
            '''
            test code
            '''
            # if len(self.ret) == 20:
            #     while not self.live_queue.empty():
            #         self.live_queue.get()
            #     self.acc_start = True
            #     break
            '''
            threshold
            '''
            if rd <= self.threshold:
                while not self.live_queue.empty():
                    self.live_queue.get()
                self.acc_start = True
                break
        print("music_detector out ", self.acc_start)
        print("[%s] %s terminated" % (p.pid, p.name))