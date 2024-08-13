import datetime
from pathlib import Path
from numpy import hanning

"""Mode"""
MODE = 'test' # live
FILE_TYPE = 'record' # midi

""" File """
DATE = datetime.date.today()

if FILE_TYPE == 'midi':
    """ MIDI 測試檔案 """
    LIVE_PATH = Path('./assessment/slow/data/live_slow_v2.wav')
    REF_PATH = Path('./assessment/ref.wav')
    ACC_PATH = Path('./assessment/acc.wav')
    FOLDER = Path(f"./{LIVE_PATH.parent.parent}/tracking_result/{LIVE_PATH.stem}/{DATE}_{REF_PATH.stem}")
else:
    if MODE == 'test':
        """真實錄音 測試檔案"""
        LIVE_PATH = Path('./real_record/beethoven/data/130_v1_blank.wav')
        # REF_PATH = Path('./real_record/beethoven/ref_data/beethoven_LKavakos_violin_25bins_clear_v2.wav')
        # ACC_PATH = Path('./real_record/beethoven/ref_data/beethoven_LKavakos_piano_25bins_clear_v2.wav')
        # REF_PATH = Path('./real_record/beethoven/ref_data/newweb_ref_v1_cut.wav')
        # ACC_PATH = Path('./real_record/beethoven/ref_data/newweb_acc_v1_cut.wav')
        REF_PATH = Path('./real_record/beethoven/ref_data/web_beethoven_conbined_violin_25bins_v2.wav')
        ACC_PATH = Path('./real_record/beethoven/ref_data/web_beethoven_conbined_piano_25bins_v2.wav')
        # REF_PATH = Path('./real_record/beethoven/ref_data/web_ref_25bins.wav')
        # ACC_PATH = Path('./real_record/beethoven/ref_data/web_acc_25bins.wav')
        # mode: test
        FOLDER = Path(f"./{REF_PATH.parent.parent}/tracking_result/{REF_PATH.stem}/{DATE}_{LIVE_PATH.stem}") 
    else:
        # mode: live
        REF_PATH = Path('./real_record/beethoven/ref_data/web_beethoven_conbined_violin_25bins_v2.wav')
        ACC_PATH = Path('./real_record/beethoven/ref_data/web_beethoven_conbined_piano_25bins_v2.wav')
        FOLDER = Path(f"./{REF_PATH.parent.parent}/tracking_result/{REF_PATH.stem}/{DATE}_{REF_PATH.stem}")
FOLDER.mkdir(parents=True, exist_ok=True)


""" Feature """
SAMPLE_RATE = 44100
HALF_SEC_FRAME = int(0.5*SAMPLE_RATE)
WINDOW_SIZE = int(0.046*SAMPLE_RATE)
NFFT = WINDOW_SIZE
HOP_SIZE = int(0.020*SAMPLE_RATE) # 20ms
FRAME_SIZE = 84 # freq band
LOW_WINDOWS_SIZE = 30
LOW_HOP_SIZE = 15
LOW_WIN = hanning(LOW_WINDOWS_SIZE)

""" Stream """
STREAM_BUFFER = WINDOW_SIZE
CHANNEL = 1
LIVE_END_COUNT = int(3*SAMPLE_RATE/WINDOW_SIZE)

""" Music detector """
MEAN_AMPLITUDE_THRESHOLD = 0.01
RMS_THRESHOLD = 0.01
MAX_ADJUST_MAG = 1.7
MIN_ADJUST_MAG = 0.7
DTW_COST_THRESHOLD = 2000

""" Music Trackers """
RPE_SEARCH_SEC = 9
RPE_SEARCH_N = int(RPE_SEARCH_SEC*SAMPLE_RATE/HOP_SIZE/LOW_HOP_SIZE)
RPE_SIMILARITY_THRESHOLD = 0.95
RPE_MAX_RUN = 3

ODTW_SEARCH_SEC = 9
ODTW_SEARCH_C = int(ODTW_SEARCH_SEC*SAMPLE_RATE/HOP_SIZE)
ODTW_BT_RANGE = int(15*SAMPLE_RATE/HOP_SIZE)
MAX_RUN = 3

HALF_SEC_HIGH_FEATURE = int(0.5*SAMPLE_RATE/HOP_SIZE)
HALF_SEC_LOW_FEATURE = int(HALF_SEC_HIGH_FEATURE/LOW_HOP_SIZE)
