import librosa
import numpy as np
# import cfg
from collections import defaultdict
# import pyaudio
import time
from multiprocessing.managers import BaseManager
from config import CHANNEL, STREAM_BUFFER, SAMPLE_RATE
import collections
from pydub import AudioSegment

from pyaudio import PyAudio, paFloat32, paComplete, paContinue

from logger import getmylogger
log = getmylogger(__name__)

INPUT_DEVICE_KEYWORD = '麥克風'
OUTPUT_DEVICE_KEYWORD = 'Realtek' # 'Focusrite USB Audio'

class Stream:
    def __init__(self, mode, live_queue, output_queue=None, test_data=None) -> None:
        self.pa = PyAudio()
        # self.test_data = None
        # self.live_queue = live_q # input queue
        # self.live = live # full test live
        # # self.live_record = np.zeros(cfg.STREAM_BUFFER, dtype=np.float32) # record online live
        # self.live_record = np.zeros(STREAM_BUFFER, dtype=np.float32) # record online live
        
        self.iodevice = self.loadIODevice()
        if mode == 'test':
            self.__initializeTestData(test_data, live_queue, output_queue)
        else:
            self.__initializeLiveData(live_queue, output_queue)
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
        self.live_record = np.concatenate((self.live_record, livedata))
        self.live_queue.put((livedata, len(livedata)))
        if len(self.output_queue) > 0:
            output_segment = self.output_queue.popleft()
            self.output_record = np.concatenate((self.output_record, output_segment))
            if output_segment is None:
                return (b'', paComplete)
            return (output_segment, paContinue)
        else:
            self.output_record = np.concatenate((self.output_record, self.mute_data))
            return (self.mute_data, paContinue)
    
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
            log.info(f"frame_count: {frame_count}")
            self.test_data = self.live[:frame_count]
            self.live_queue.put((self.test_data, len(self.test_data)))
            self.live = self.live[frame_count:]
            self.live_record = np.concatenate((self.live_record, self.test_data))
            # return (self.mute_data, paContinue)
            if len(self.output_queue) > 0:
                output_segment = self.output_queue.popleft()
                if output_segment is None:
                    print("output_segment is None")
                    return (b'', paComplete)
                self.output_record = np.concatenate((self.output_record, output_segment))
                return (output_segment, paContinue)
            else:
                self.output_record = np.concatenate((self.output_record, self.mute_data))
                return (self.mute_data, paContinue)
    
    def __initializeLiveData(self, live_queue, output_queue):
        """Initial real live tracking parameters

        Args:
            queue (_type_): _description_
        """
        self.live_queue = live_queue
        self.output_queue = output_queue
        self.live_record = np.zeros(0, dtype=np.float32) # record online live
        self.output_record = np.zeros(0, dtype=np.float32) # record online output
        self.mute_data = np.zeros(STREAM_BUFFER, dtype=np.float32)
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
        
    def __initializeTestData(self, test_live, live_queue, output_queue):
        """Initial test data tracking parameters

        Args:
            test_live (_type_): _description_
            queue (_type_): _description_
        """
        self.live = test_live # full test live
        self.frame, self.pre_frame = 0, 0
        self.test_data = None
        self.live_queue = live_queue
        self.output_queue = output_queue
        self.live_record = np.zeros(0, dtype=np.float32) # record online live
        self.output_record = np.zeros(0, dtype=np.float32) # record online output
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
        self.stream.start_stream()

if __name__ == '__main__':
    stream = Stream()
    stream.listIODevice()