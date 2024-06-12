import librosa
import numpy as np
import cfg
from collections import defaultdict
# import pyaudio
import time
from multiprocessing.managers import BaseManager
from cfg import CHANNEL, STREAM_BUFFER, SAMPLE_RATE
import collections
from pydub import AudioSegment

from pyaudio import PyAudio, paFloat32, paComplete, paContinue

from logger import getmylogger
log = getmylogger(__name__)

INPUT_DEVICE_KEYWORD = '麥克風'
OUTPUT_DEVICE_KEYWORD = 'Realtek' # 'Focusrite USB Audio'

class Stream:
    def __init__(self, mode, queue, acc=None, output_queue=None, test_data=None) -> None:
        self.pa = PyAudio()
        # self.test_data = None
        # self.live_queue = live_q # input queue
        # self.live = live # full test live
        # # self.live_record = np.zeros(cfg.STREAM_BUFFER, dtype=np.float32) # record online live
        # self.live_record = np.zeros(STREAM_BUFFER, dtype=np.float32) # record online live
        
        self.iodevice = self.loadIODevice()
        if mode == 'test':
            self.__initializeTestData(test_data, queue)
        else:
            self.__initializeLiveData(queue, acc, output_queue)
        log.info("Data initialization completed")
        # self.stream = self.pa.open(format = paFloat32,
        #                         #    channels=cfg.CHANNEL,
        #                            channels=CHANNEL,
        #                            input_device_index = self.iodevice[0],
        #                            output_device_index = self.iodevice[1],
        #                         #    rate=cfg.SAMPLE_RATE,
        #                            rate=SAMPLE_RATE,
        #                            output=True,
        #                            input=True,
        #                            stream_callback=self.__callback,
        #                         #    frames_per_buffer = cfg.STREAM_BUFFER)
        #                            frames_per_buffer = STREAM_BUFFER)

        # self.mute_data = np.zeros(STREAM_BUFFER, dtype=np.float32)
        # self.mute_data = np.array([0.00001]*2048, dtype=np.float32)
        
        # self.stream.stop_stream()
        
        # self.path = []
        
        # self.frame,self.pre_frame = 0,0
    
    def listIODevice(self):
        """List IO Device information
        """
        for device in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(device)
            log.info(f"IO Device: {device_info['index'], device_info['name'], device_info['maxInputChannels'], device_info['maxOutputChannels']}")
            
    def loadIODevice(self):
        """Load IO Device with keywords.

        Returns:
            _type_: _description_
        """
        dinput = None
        doutput = None
        for device in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(device)
            if dinput == None and device_info['name'].find(INPUT_DEVICE_KEYWORD) >=0 and device_info['maxInputChannels'] > 0:
                dinput = (device, device_info['name'])
            if doutput == None and device_info['name'].find(OUTPUT_DEVICE_KEYWORD) >= 0 and device_info['maxOutputChannels'] > 0:
                doutput = (device, device_info['name'])
        
        try:
            if not dinput or not doutput:
                assert False, 'IO device has None !'
            
            log.info(f"Use IO Device: {dinput[1], doutput[1]}")
            return [dinput[0], doutput[0]]
        except AssertionError as msg:
            log.error(f"IO device failed to load: {msg}")
        except:
            log.critical(f"Unknown Error !")

    def __liveCallback(self, livedata, frame_count, time_info, flag):
        """Get real live from input device to tracking.

        Args:
            livedata (_type_): _description_
            frame_count (_type_): _description_
            time_info (_type_): _description_
            flag (_type_): _description_
        """
        livedata = np.frombuffer(livedata, dtype=np.float32)
        self.live_queue.put((livedata, len(livedata)))
        if len(self.acc_queue) == 0:
            self.acc_data = np.zeros(frame_count, dtype=np.float32)
            return (self.acc_data, paContinue)
        else:
            acc_position = self.acc_queue.peek_last()
            if acc_position is None:
                self.live_queue.put(None)
                return (b'', paComplete)
            else:
                # return acc
                pass
    
    def __callback(self, livedata, frame_count, time_info, flag):
        """Use test data to tracking.

        Args:
            livedata (_type_): _description_
            frame_count (_type_): _description_
            time_info (_type_): _description_
            flag (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        if len(self.live) == 0:
            log.info("Test live has ended.")
            self.live_queue.put((self.mute_data, len(self.mute_data)))
            return (b'', paComplete)
        else:
            self.test_data = self.live[:frame_count]
            self.live_queue.put((self.test_data, len(self.test_data)))
            self.live = self.live[frame_count:]
        self.live_record = np.concatenate((self.live_record, self.test_data))
        return (self.mute_data, paContinue)
    
    def __initializeLiveData(self, queue, acc, output_queue):
        """Initial real live tracking parameters

        Args:
            queue (_type_): _description_
        """
        self.live_queue = queue
        self.acc_queue = output_queue
        self.acc = acc
        self.live_record = np.zeros(STREAM_BUFFER, dtype=np.float32) # record online live
        self.stream = self.pa.open(format = paFloat32,
                                   channels=CHANNEL,
                                   input_device_index = self.iodevice[0],
                                   output_device_index = self.iodevice[1],
                                   rate=SAMPLE_RATE,
                                   output=True,
                                   input=True,
                                   stream_callback=self.__liveCallback,
                                   frames_per_buffer = STREAM_BUFFER)
        self.stream.stop_stream()
        
    def __initializeTestData(self, test_live, queue):
        """Initial test data tracking parameters

        Args:
            test_live (_type_): _description_
            queue (_type_): _description_
        """
        self.live = test_live # full test live
        self.live_queue = queue
        self.test_data = None
        self.frame, self.pre_frame = 0, 0
        self.live_record = np.zeros(STREAM_BUFFER, dtype=np.float32) # record online live
        self.mute_data = np.zeros(STREAM_BUFFER, dtype=np.float32)
        self.stream = self.pa.open(format = paFloat32,
                                   channels=CHANNEL,
                                   input_device_index = self.iodevice[0],
                                   output_device_index = self.iodevice[1],
                                   rate=SAMPLE_RATE,
                                   output=True,
                                   input=True,
                                   stream_callback=self.__callback,
                                   frames_per_buffer = STREAM_BUFFER)
        self.stream.stop_stream()
        log.info(f"Test live: {self.live.shape, self.live_queue.qsize()}")
        
    def start(self):
        """
        Start streaming
        """
        # if is_live:
        #     self.__initializeLiveData(queue)
        # else:
        #     self.__initializeTestData(test_data, queue)
        
        self.stream.start_stream()

if __name__ == '__main__':
    stream = Stream()
    stream.listIODevice()