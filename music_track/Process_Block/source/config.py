# frame reader
SAMPLE_RATE = 44100
WINDOW_SIZE = 2048
FRAME_SIZE = 84 # freq band
HOP_SIZE = 882 # 20ms

STFT = {
    'nfft': 2048,
    'hop': 512,
    'window': 2048
}
# odtw
MAX_RUN = 3
DIAG_COST_FACTOR=1
SECS_TO_TICKS = 1 * 1000 * 1000 * 10