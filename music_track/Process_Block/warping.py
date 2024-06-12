from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
import numpy as np
import time
from .source import ODTW
from .util import getLiveDataBySec

class ODTWProcess(Process):
    def __init__(self, *args, **kwargs):
        self.live_queue = kwargs.pop('live_queue') # [(livedata, ref[start_frame:start_frame+9sec]), ...]
        self.odtw_queue = kwargs.pop('odtw_queue') # [(livedata, ref[start_frame:start_frame+9sec]), ...]
        self.candidate_queue = kwargs.pop('candidate_queue') # [(livedata, ref[start_frame:start_frame+9sec]), ...]
        self.share_data = kwargs.pop('share_data')
        self.ref_data = self.share_data['ref_data']
        self.odtw = ODTW()
        self.ret = []
        super().__init__(*args, **kwargs)
    
    def run(self):
        p = current_process()
        print("New process -> [%s] %s" % (p.pid, p.name))
        while not self.share_data['acc_is_stop']:
            start = time.time()
            live_data = self.live_queue.get()
            live = getLiveDataBySec(live_data, sec=9)
            params = self.odtw_queue.get()
            print("ODTW start!")
            # normal (slower)
            # ret = []
            # for p in params:
            #     ret.append(self.odtw.run(live, p[0]))
            
            # pool (faster)
            odtw_param = []
            for param in params:
                odtw_param.append((live, self.ref_data[param[0]:param[1]]))
            pool = Pool(7)
            ret = pool.starmap(self.odtw.run, odtw_param)
            min_idx = ret.index(min(ret, key=lambda a: a[1]))
            print(params)
            print(ret[min_idx][0][-1]) # now frame
            print(ret[min_idx][0][-1][1]) # now frame
            print(params[min_idx][0]) # now frame
            print(ret[min_idx][0][-1][1]+params[min_idx][0]) # now frame
            pool.close() # close只是關閉pool, 但已開啟的process還是會繼續執行  
            pool.join()
            
            self.candidate_queue.put(ret)
            
            print("ODTW block end: ", time.time()-start)
            # time.sleep(0.01)
        print("[%s] %s terminated" % (p.pid, p.name))