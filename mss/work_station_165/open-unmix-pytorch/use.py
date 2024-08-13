from openunmix.predict import separate
import torchaudio
import soundfile as sf
import os
from pathlib import Path
from museval.metrics import bss_eval
import numpy as np
from collections import defaultdict
# run N250 and N2000 data

def compute_uSDR(
        y_hat: np.ndarray,
        y_tgt: np.ndarray,
        delta: float = 1e-7,
) -> float:
    """
    Computes SDR metric as in https://arxiv.org/pdf/2108.13559.pdf.
    Taken and slightly rewritten from
    https://github.com/AIcrowd/music-demixing-challenge-starter-kit/blob/master/evaluator/music_demixing.py
    """
    # compute SDR for one song
    num = np.sum(np.square(y_tgt), axis=(1, 2))
    den = np.nansum(np.square(y_tgt - y_hat), axis=(1, 2))
    # print(np.where(np.isnan(np.square(y_tgt - y_hat))))
    # print(den)
    num += delta
    den += delta
    # print(num, den)
    return 10 * np.log10(num / den)

root_path = './aligned_dataset/test'
song_folder = os.listdir(root_path)
model_path = Path('./open-unmix-data-limit')
metrics = defaultdict(list)
for song_path in song_folder:
    mix_audio = f'{root_path}/{song_path}/mixture.wav'
    # violin_audio = f'{root_path}/{song_path}/violin.wav'
    # piano_audio = f'{root_path}/{song_path}/piano.wav'
    print(song_path)
    mix, rate = torchaudio.load(mix_audio)
    # vt, rate = torchaudio.load(violin_audio)
    # pt, rate = torchaudio.load(piano_audio)
    estimates = separate(mix, rate=rate,
                        model_str_or_path=str(model_path),
                        targets=['violin', 'piano'])
    # 計算uSDR指標
    # vhat = estimates['violin'][0][0].unsqueeze(0)
    # phat = estimates['piano'][0][0].unsqueeze(0)
    # vhat = vhat.T.unsqueeze(0).numpy()
    # vt = vt.T.unsqueeze(0).numpy()
    # phat = phat.T.unsqueeze(0).numpy()
    # pt = pt.T.unsqueeze(0).numpy()
    # uSDR = compute_uSDR(
    #     vhat,
    #     vt
    # )
    # metrics['violin_uSDR'].append(uSDR)
    # print("violin_uSDR", uSDR)
    # uSDR = compute_uSDR(
    #     phat,
    #     pt
    # )
    # metrics['piano_uSDR'].append(uSDR)
    # print("piano_uSDR", uSDR)
    
    # 儲存audio results
    for target, estimate in estimates.items():
        os.umask(0)
        output_dir = Path(os.path.join(f"./estimates/{song_path}/{model_path.name}/{target}"))
        output_dir.mkdir(0o777, exist_ok = True, parents = True)
        print(target, estimate[0][0].T.shape)
        sf.write(
            str(output_dir / Path(target).with_suffix('.wav')),
            estimate[0][0].T,
            44100)
    
    
# metrics['violin_uSDR'] = np.array(metrics['violin_uSDR'])
# metrics['piano_uSDR'] = np.array(metrics['piano_uSDR'])
# for target in metrics:
#     print(f"Metric - {target}, mean - {metrics[target].mean():.3f}")
    


