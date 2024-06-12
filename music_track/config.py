from numpy import hanning
from pathlib import Path
import datetime

""" File """
DATE = datetime.date.today()
# LIVE_PATH = Path('./assessment/slow/data/live_slow_v2.wav')
# REF_PATH = Path('./assessment/ref.wav')
# ACC_PATH = Path('./assessment/acc.wav')
# FOLDER = Path(f"./{LIVE_PATH.parent.parent}/tracking_result/{LIVE_PATH.stem}/{DATE}_{REF_PATH.stem}")

LIVE_PATH = Path('./real_record/beethoven/data/newweb_v3_blank.wav')
# REF_PATH = Path('./real_record/beethoven/ref_data/beethoven_LKavakos_violin_25bins_clear_v2.wav')
# ACC_PATH = Path('./real_record/beethoven/ref_data/beethoven_LKavakos_piano_25bins_clear_v2.wav')
REF_PATH = Path('./real_record/beethoven/ref_data/ref_newweb_v1.wav')
# REF_PATH = Path('./real_record/beethoven/ref_data/web_beethoven_conbined_violin_25bins_v2.wav')
ACC_PATH = Path('./real_record/beethoven/ref_data/web_beethoven_conbined_piano_25bins_v2.wav')
# REF_PATH = Path('./real_record/beethoven/ref_data/ref_web_25bins.wav')
# ACC_PATH = Path('./real_record/beethoven/ref_data/acc_web_25bins.wav')
FOLDER = Path(f"./{REF_PATH.parent.parent}/tracking_result/{REF_PATH.stem}/{DATE}_{LIVE_PATH.stem}") # mode: test
# FOLDER = Path(f"./{REF_PATH.parent.parent}/tracking_result/{REF_PATH.stem}/{DATE}_{REF_PATH.stem}") # mode: live
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
STREAM_BUFFER = 2048
CHANNEL = 1

""" Music detector """
MEAN_AMPLITUDE_THRESHOLD = 0.01 # 0.015
RMS_THRESHOLD = 0.01
MAX_ADJUST_MAG = 1.7
MIN_ADJUST_MAG = 0.7
DTW_COST_THRESHOLD = 2000 # 1620

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

# SPEED_WT = 0.04
