import pyaudio
import numpy as np
import time
from config import CHANNEL, STREAM_BUFFER, SAMPLE_RATE, SPEED_WT
import librosa

class TestStream:
    def __init__(self, live, acc, live_q, acc_q, share_acc_record) -> None:
        self.pa = pyaudio.PyAudio()
        self.test_data = None
        self.live_queue = live_q # input queue
        self.acc_queue = acc_q # output deque
        self.share_acc_record = share_acc_record
        self.live = live # full test live
        # self.live_record = np.zeros(cfg.STREAM_BUFFER, dtype=np.float32) # record online live
        self.live_record = np.zeros(STREAM_BUFFER, dtype=np.float32) # record online live
        
        # self.live_acc = np.zeros(cfg.STREAM_BUFFER, dtype=np.float32)
        self.live_acc = np.zeros(0, dtype=np.float32)
        self.acc = acc
        self.new_acc = acc
        self.acc_frame = -1
        
        self.now_frame = 0
        
        iodevice = self.getInputAndOutputDevice()
        self.stream = self.pa.open(format=pyaudio.paFloat32,
                                #    channels=cfg.CHANNEL,
                                   channels=CHANNEL,
                                   input_device_index = iodevice[0],
                                   output_device_index = iodevice[1],
                                #    rate=cfg.SAMPLE_RATE,
                                   rate=SAMPLE_RATE,
                                   output=True,
                                   input=True,
                                   stream_callback=self.__callback,
                                #    frames_per_buffer = cfg.STREAM_BUFFER)
                                   frames_per_buffer = STREAM_BUFFER)
        self.stream.stop_stream()
        
        self.path = []
        
        self.frame,self.pre_frame = 0,0
        print("SPEED_WT", SPEED_WT)
        
    def getInputAndOutputDevice(self):
        dinput = None
        doutput = None
        for device in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(device)
            if dinput == None and device_info['name'] == '麥克風 (2- Realtek(R) Audio)' and device_info['maxInputChannels'] > 0:
                dinput = device
            if doutput == None and device_info['name'] == '喇叭 (Focusrite USB Audio)' and device_info['maxOutputChannels'] > 0:
                doutput = device
        print("Device: ", dinput, doutput)
        return [dinput, doutput]
    
    def __live_callback(self, livedata, frame_count, time_info, flag):
        livedata = np.frombuffer(livedata, dtype=np.float32)
        self.live_record = np.concatenate((self.live_record, livedata))
        self.live_queue.put(livedata)
        
        if len(self.acc_queue) == 0:
            self.acc_data = np.zeros(frame_count, dtype=np.float32)
            self.live_acc = np.concatenate((self.live_acc, self.test_acc_data))
            return (self.acc_data, pyaudio.paContinue)
        else:
            acc_position = self.acc_queue.peek_last()
            if acc_position is None:
                self.live_queue.put(None)
                return (b'', pyaudio.paComplete)
            else:
                # return acc
                pass
                
    def __callback(self, livedata, frame_count, time_info, flag):
        # self.live_queue.put((livedata,frame_count))
        # start = time.time()
        # if len(self.test_data) == 0:
        if len(self.live) == 0:
            print("no live")
            self.test_data = np.zeros(STREAM_BUFFER, dtype=np.float32)
            self.live_queue.put((self.test_data, len(self.test_data)))
        else:
            self.test_data = self.live[:frame_count]
            self.live_queue.put((self.test_data, len(self.test_data)))
            self.live = self.live[frame_count:]
        self.live_record = np.concatenate((self.live_record, self.test_data))
        
        # if self.acc_queue.empty() or len(self.test_acc_queue)==0:
        # TODO: 換用self.acc_record輸出 失敗
        if len(self.acc_queue)==0:
            # not yet
            # self.test_acc_data = self.new_acc[:frame_count]
            # self.new_acc = self.new_acc[frame_count:]
            # self.live_acc = np.concatenate((self.live_acc, self.test_acc_data))
            # return (self.test_acc_data, pyaudio.paContinue)
            print("no acc")
            self.test_acc_data = np.zeros(STREAM_BUFFER, dtype=np.float32)
            self.live_acc = np.concatenate((self.live_acc, self.test_acc_data))
            # self.acc_record = np.concatenate((self.acc_record, np.zeros(882, dtype=np.float32)))
            # return (self.test_acc_data, pyaudio.paContinue)
        else:
            # acc_position = self.acc_queue.get()
            acc_position = self.acc_queue.peek_last()
            print("stream: ", acc_position)
            if acc_position is None:
                print("acc_position is None!")
                self.live_queue.put(None)
                return (b'', pyaudio.paComplete)
            else:
                l_frame, frame = acc_position # 可能會一直重疊到(2048-882)個點?
                self.path.append((l_frame//882, frame//882))
                self.frame = frame
                
                # if self.frame != self.pre_frame:
                #     self.new_acc = self.acc[self.frame:]
                # self.pre_frame = self.frame
                
                self.new_acc = self.acc[self.frame:]
                
                self.test_acc_data = self.new_acc[:frame_count]
                # if len(self.share_acc_record['acc_record']) > STREAM_BUFFER:
                #     print("use share acc!")
                #     self.test_acc_data = self.share_acc_record['acc_record'][-STREAM_BUFFER:]
                self.new_acc = self.new_acc[frame_count:]
                self.live_acc = np.concatenate((self.live_acc, self.test_acc_data))
                # self.live_acc = np.concatenate((self.live_acc, self.acc[self.frame:self.frame+882]))
            # if not self.acc_record_queue.empty():
            #     print(self.acc_record_queue.qsize())
            #     pos = self.acc_record_queue.get()
            #     self.acc_record = np.concatenate((self.acc_record, self.acc[pos:pos+882]))
        # print("stream per time: ", time.time()-start)
        return (self.test_acc_data, pyaudio.paContinue)
                
        # if self.acc_frame == -1:
        #     # no new pos
        #     self.test_acc_data = self.new_acc[:frame_count]
        #     self.new_acc = self.new_acc[frame_count:]
        # else:
        #     # new pos
        #     print("streaming ", self.acc_frame)
        #     if abs(self.acc_frame-self.now_frame) <= 441000:
        #         self.new_acc = self.acc[self.acc_frame:]
        #         self.now_frame = self.acc_frame
        #     # self.new_acc = self.acc[self.acc_frame:]
        #     self.test_acc_data = self.new_acc[:frame_count]
        #     self.new_acc = self.new_acc[frame_count:]
        #     self.acc_frame = -1
        # self.now_frame+=frame_count
        
        # self.live_acc = np.concatenate((self.live_acc, self.test_acc_data))
        
        # # if self.acc_frame != -1:
        # #     print("streaming ", self.acc_frame)
        # #     self.acc_frame = -1
        # return (self.test_acc_data, pyaudio.paContinue)
    
    def start(self):
        self.stream.start_stream()
        startTime = time.time()
        print("start streaming: ", startTime)