from pathlib import Path
from pydub import AudioSegment
import subprocess
import random
from tqdm import tqdm

open_violin_dataset = Path('../dataset/violin')
open_piano_dataset = Path('../dataset/piano')
open_dataset_path = Path('./dataset/train')
open_dataset_path.mkdir(parents=True, exist_ok=True)

def convert_audio_suffix(tracks_path, output_path=None, output_suffix='.wav'):
    for track in tracks_path:
        subprocess.call(['ffmpeg', '-i', f'{track}', f'{output_path+track.stem+output_suffix}'])

def random_mixing_track(track_len, alignment_len):
    rdlist = [num for num in range(track_len)]
    random.shuffle(rdlist)
    rdlist = rdlist[:alignment_len]
    return rdlist

violin_tracks = list(open_violin_dataset.glob('**/*.wav'))
piano_tracks = list(open_piano_dataset.glob('**/*.wav'))
rdlist = random_mixing_track(len(piano_tracks), len(violin_tracks))

for vtrack_idx, ptrack_idx in enumerate(tqdm(rdlist)):
    # read audio file
    violin = AudioSegment.from_wav(violin_tracks[vtrack_idx])
    piano = AudioSegment.from_wav(piano_tracks[ptrack_idx])
    
    # set to mono channels
    violin = violin.set_channels(1) if violin.channels == 2 else violin
    piano = piano.set_channels(1) if piano.channels == 2 else piano
    
    # set to the same duration
    if violin.duration_seconds > piano.duration_seconds:
        violin = violin[:len(piano)]
    else:
        piano = piano[:len(violin)]
    
    # set to the same dBFS
    dbfs_range = violin.dBFS - piano.dBFS
    if dbfs_range > 0:
        piano += dbfs_range
    else:
        piano -= dbfs_range
    
    # overlay
    mixture = piano.overlay(violin)
    
    # write audio file
    output_path = Path(f'{open_dataset_path}/{violin_tracks[vtrack_idx].stem}_mix_{piano_tracks[ptrack_idx].stem}')
    try:
        output_path.mkdir(parents=True, exist_ok=False)
        violin.export(f"{output_path}/violin.wav", format="wav")
        piano.export(f"{output_path}/piano.wav", format="wav")
        mixture.export(f"{output_path}/mixture.wav", format="wav")
    except:
        print("This mixture has exist!")
        continue