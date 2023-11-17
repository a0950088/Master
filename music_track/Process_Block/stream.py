import pyaudio
import numpy as np
import time
import config as cfg
import librosa

class TestStream:
    def __init__(self, acc, share_data) -> None:
        test_live, _ = librosa.core.load("./music/test/summer3rd_testlive.wav", sr=cfg.SAMPLE_RATE)
        # testing
        self.test_live = np.concatenate((np.zeros(int(cfg.ONE_BUFFER_SEC*3), dtype=np.float32), test_live))
        
        self.pa = pyaudio.PyAudio()
        self.live_record = np.zeros(0, dtype=np.float32)
        self.live_acc = np.zeros(cfg.STREAM_BUFFER, dtype=np.float32)
        self.acc = acc
        iodevice = self.getInputAndOutputDevice()
        self.stream = self.pa.open(format=pyaudio.paFloat32,
                                   channels=cfg.CHANNEL,
                                   input_device_index = iodevice[0],
                                   output_device_index = iodevice[1],
                                   rate=cfg.SAMPLE_RATE,
                                   output=True,
                                   input=True,
                                   stream_callback=self.__callback,
                                   frames_per_buffer = cfg.STREAM_BUFFER)
        self.share_data = share_data

    def getInputAndOutputDevice(self):
        dinput = None
        doutput = None
        for device in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(device)
            if dinput == None and device_info['name'] == '麥克風 (2- Realtek(R) Audio)' and device_info['maxInputChannels'] > 0:
                dinput = device
            if doutput == None and device_info['name'] == '喇叭 (Focusrite USB Audio)' and device_info['maxOutputChannels'] > 0:
                doutput = device
        print("Device: ",dinput,doutput)
        return [dinput, doutput]
    
    def __callback(self, livedata, frame_count, time_info, flag):
        # input live
        '''
        livedata = np.frombuffer(livedata, dtype=np.float32)
        self.live_record = np.concatenate((self.live_record, livedata))
        '''
        
        # test live
        testlive_data = self.test_live[:frame_count]
        self.test_live = self.test_live[frame_count:]
        self.live_record = np.concatenate((self.live_record, testlive_data))
        # accompaniment        
        if not self.share_data['acc_is_stop']:
            self.live_acc = self.acc[:int(cfg.STREAM_BUFFER*self.share_data['factor'])]
            self.acc = self.acc[int(cfg.STREAM_BUFFER*self.share_data['factor']):]
        # return (self.live_acc, pyaudio.paContinue)
        return (testlive_data, pyaudio.paContinue)
    
    def start(self):
        self.stream.start_stream()
        startTime = time.time()
        print("start streaming: ", startTime)