from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
from source.odtw import ODTW
import numpy as np
import time
import scipy.io.wavfile
class DMA(Process):
    def __init__(self, *args, **kwargs):
        ref = kwargs.pop('ref')
        live_queue = kwargs.pop('live_queue')
        rpe_queue = kwargs.pop('rpe_queue')
        acc_queue = kwargs.pop('acc_queue')
        share_acc_record = kwargs.pop('share_acc_record')
        self.status = kwargs.pop('status')
        self.odtw = ODTW(ref, live_queue, rpe_queue, acc_queue, share_acc_record)
        self.acc = kwargs.pop('acc')
        # self.rpe_queue = rpe_queue
        
        super().__init__(*args, **kwargs)
    
    def run(self):
        p = current_process()
        print("New process -> [%s] %s" % (p.pid, p.name))
        self.odtw.run()
        offline_acc_record = np.zeros(0, dtype=np.float32)
        for i,j in self.odtw.prime_path:
            if j==0:
                pass
            else:
                offline_acc_record = np.concatenate((offline_acc_record, self.acc[(j-1)*882:j*882]))
        scipy.io.wavfile.write('off_acc.wav', 44100, np.array(offline_acc_record, dtype=np.float32))
        self.odtw.optimalWarpingPath()
        self.odtw.draw()
        # return (frame, cost)
        print("[%s] %s terminated" % (p.pid, p.name))