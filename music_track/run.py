import librosa
from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
from Process_Block import MusicDetector, RoughEstimator, ODTWProcess, DecisionMaker, TestStream
import time

SAMPLE_RATE = 44100

if __name__ == "__main__":
    # set_start_method('spawn')
    # freeze_support()
    
    # load audio file
    ref, _ = librosa.core.load("./music/test/summer3rd_violin.wav", sr=SAMPLE_RATE)
    acc, _ = librosa.core.load("./music/test/summer3rd_piano.wav", sr=SAMPLE_RATE)
    
    # create shared memory
    manage = Manager()
    live_queue = manage.Queue(maxsize=5)
    odtw_queue = manage.Queue()
    candidate_queue = manage.Queue()
    # acc_queue = manage.Queue()
    share_data = manage.dict({'acc_is_stop': True, 'factor': 1, 'ref_data': ref})
    md_proc = MusicDetector(queue=live_queue, share_data=share_data)
    re_proc = RoughEstimator(live_queue=live_queue, odtw_queue=odtw_queue, share_data=share_data)
    ot_proc = ODTWProcess(live_queue=live_queue, odtw_queue=odtw_queue, candidate_queue=candidate_queue, share_data=share_data)
    dm_proc = DecisionMaker(candidate_queue=candidate_queue, share_data=share_data)
    print("initial process")
    
    
    # main process
    parent_p = current_process()
    print("Main process -> [%s] %s" % (parent_p.pid, parent_p.name))
    testst = TestStream(acc, share_data=share_data)
    time.sleep(1)
    live_queue.put(testst.live_record)
    print("init live: ", testst.live_record.shape)
    
    # initial sub process
    print()
    print("==========detecting loop=========")
    # detecting loop
    md_proc.start()
    while testst.stream.is_active():
        md_proc.join(timeout=0)
        if md_proc.is_alive():
            print("alive")
            if live_queue.full():
                live_queue.get_nowait()
            print(testst.live_record.shape)
            live_queue.put(testst.live_record)
            time.sleep(0.1)
        else:
            print("now queue: ", live_queue.qsize())
            md_proc.join()
            break
    # while not live_queue.empty():
    #     print(live_queue.get().shape)
    md_proc.join()
    # parent_p.join()
    print()
    print("==========tracking loop=========")
    # tracking loop
    share_data['acc_is_stop'] = False
    re_proc.start()
    ot_proc.start()
    dm_proc.start()
    while testst.stream.is_active():
        re_proc.join(timeout=0)
        ot_proc.join(timeout=0)
        dm_proc.join(timeout=0)
        if dm_proc.is_alive() and ot_proc.is_alive() and re_proc.is_alive():
            if live_queue.full():
                live_queue.get_nowait()
            live_queue.put(testst.live_record)
            if len(testst.acc) == 0:
                share_data['acc_is_stop'] = True
            # print("main 2: ", live_queue.qsize())
            time.sleep(0.05)
        else:
            print("now queue 2: ", live_queue.qsize())
            break
    
    re_proc.join()
    ot_proc.join()
    dm_proc.join()
    
    
    time.sleep(5)