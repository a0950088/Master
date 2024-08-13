import museval
import os
from pathlib import Path
import shutil
import numpy as np
from statistics import mean


'''
把test dataset的audio results存起來後計算cSDR指標

uSDR指標在./work_station_165/open-unmix-pytorch/use.py 中計算

N250(42min) Violin {'cSDR': 2.0485406250000002, 'SIR': nan, 'ISR': 6.5276590625, 'SAR': 1.3744415625} / uSDR: 1.72
N250(42min) Piano {'cSDR': 10.9480521875, 'SIR': nan, 'ISR': 14.308698750000001, 'SAR': 11.501214999999998} / uSDR: 10.974
N2000(5.5 hr) Violin {'cSDR': 5.8110800000000005, 'SIR': nan, 'ISR': 9.9389065625, 'SAR': 5.6762540625} / uSDR: 5.666
N2000(5.5 hr) Piano {'cSDR': 13.514704374999999, 'SIR': nan, 'ISR': 22.879948437499998, 'SAR': 13.59446125} / uSDR: 14.124
'''

reference_root = "./test"
estimate_root = "./estimates_v2"
song_list = os.listdir(reference_root)
model = 'open-unmix_N2000'
target = 'piano'
metrics = {
    'SDR':[], 
    'SIR':[], 
    'ISR':[], 
    'SAR':[]
}

# for song_path in song_list:
#     result = museval.eval_dir(f"{reference_root}/{song_path}/{target}",
#                               f"{estimate_root}/{song_path}/{model}/{target}", win=1)
#     for t in result.scores['targets']:
#         for m in metrics.keys():
#             res = result.frames_agg([np.float64(f['metrics'][m])
#                         for f in t['frames']])
#             metrics[m].append(res)
#         print(metrics)
# for m in metrics.keys():
#     metrics[m] = np.mean(np.array(metrics[m]))
# print(metrics)

for song_path in song_list:
    result = museval.eval_dir(f"{reference_root}/{song_path}/{target}",
                              f"{estimate_root}/{song_path}/{model}/{target}", win=1)
    print(result.shape)
    cSDR = np.nanmedian(result)
    print(cSDR)
    metrics['SDR'].append(cSDR)

metrics['SDR'] = np.mean(np.array(metrics['SDR']))
print(metrics['SDR'])
    

