from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
import numpy as np
import time

class DecisionMaker(Process):
    def __init__(self, *args, **kwargs):
        self.candidate_queue = kwargs.pop('candidate_queue') # [(livedata, ref[start_frame:start_frame+9sec]), ...]
        # self.acc_queue = kwargs.pop('acc_queue') # [(livedata, ref[start_frame:start_frame+9sec]), ...]
        self.share_data = kwargs.pop('share_data')
        self.ref_data = self.share_data['ref_data']
        self.ret = []
        super().__init__(*args, **kwargs)

    def run(self):
        p = current_process()
        print("New process -> [%s] %s" % (p.pid, p.name))
        while not self.share_data['acc_is_stop']:
            # start = time.time()
            if self.candidate_queue.empty():
                self.share_data['factor'] = 1
            else:
                candidate = self.candidate_queue.get()
                print("dm: ", candidate)
                self.share_data['factor'] = 1.3
            # time.sleep(0.05)
            # print("dm end: ", time.time()-start)
        print("[%s] %s terminated" % (p.pid, p.name))