from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
import numpy as np
import time
from .util import liveAndRefDist, getLiveDataBySec
import config as cfg

class RoughEstimator(Process):
    def __init__(self, *args, **kwargs):
        self.live_queue = kwargs.pop('live_queue')
        self.odtw_queue = kwargs.pop('odtw_queue')
        self.share_data = kwargs.pop('share_data')
        self.ref_data = self.share_data['ref_data']
        self.est_sec = int(cfg.ONE_BUFFER_SEC*9)
        self.ret = []
        super().__init__(*args, **kwargs)
    
    def run(self):
        p = current_process()
        print("New process -> [%s] %s" % (p.pid, p.name))
        while not self.share_data['acc_is_stop']:
            start = time.time()
            print("rough estimator start!")
            live_data = self.live_queue.get()
            live_data = getLiveDataBySec(live_data, sec=9)
            pool = Pool(2)
            params = []
            for i in range(0, len(self.ref_data), int(self.est_sec*0.5)):
                if len(self.ref_data[i:i+self.est_sec]) == self.est_sec:
                    ref = self.ref_data[i:i+self.est_sec]
                    # params.append((live_data, ref, i)) # live_data, ref_data[i:i+est_sec], start_frame count
                    params.append((live_data, ref, i, i+self.est_sec)) # live_data, ref_data[i:i+est_sec], start_frame count
                else: 
                    ref = self.ref_data[len(self.ref_data)-self.est_sec:]
                    # params.append((live_data, ref, len(self.ref_data)-self.est_sec))
                    params.append((live_data, ref, len(self.ref_data)-self.est_sec, len(self.ref_data)))

            ret = pool.starmap(liveAndRefDist, params)
            ret = sorted(ret, key=lambda r: r[0])[:4] # get 4 highest similarity data
            pool.close() # close只是關閉pool, 但已開啟的process還是會繼續執行
            pool.join()
            # re_queue.put(list(zip(*ret))[1])
            # self.ret = ret
            ret = list(zip(*ret))
            possible_odtw = list(zip(ret[1], ret[2]))
            # print(possible_odtw)
            # possible_odtw = "data"
            self.odtw_queue.put(possible_odtw)
            print("rough estimator block end: ", time.time()-start)
            # time.sleep(0.1)
        print("[%s] %s terminated" % (p.pid, p.name))