import os
import random
import mido
import numpy as np

'''
slow: 90-120 bpm
normal: 115-145 bpm
fast: 135-175 bpm
accelerate: 80-160 bpm
'''
def randomParameter():
    bpm = random.randint(135, 175) # set random bpm range
    change_tempo_prob = random.randint(0, 1) # 50%
    
    return bpm, change_tempo_prob

def bpmToMidoTempo(bpm):
    return 6*1e7/bpm

def midoTempoToBpm(tempo):
    return 6*1e7/tempo

midi_path = './assessment/beethoven_spring_25bin_test.mid'
mid = mido.MidiFile(midi_path, clip=True)

mo = mido.MidiFile(type = 1)# output file
new_track = mido.MidiTrack()

violin_track = mid.tracks[0]

accelerate_bpm = np.linspace(start=81, stop=160, num=165)
idx = 0
# write new midi file
for msg in violin_track:
    
    # accelerate fixed range bpm
    # if isinstance(msg, mido.MetaMessage) and msg.type == 'set_tempo':
    #     bpm, _ = randomParameter()
    #     msg.tempo = round(bpmToMidoTempo(80))
    # if isinstance(msg, mido.Message) and msg.type == 'note_on' and msg.velocity>0 and msg.time>0:
    #     if idx < len(accelerate_bpm):
    #         new_msg = mido.MetaMessage('set_tempo', tempo=round(bpmToMidoTempo(accelerate_bpm[idx])), time=0)
    #         idx+=1
    #         new_track.append(new_msg)
    # new_track.append(msg)
    # print("msg: ", msg.type, vars(msg), isinstance(msg, mido.MetaMessage))
    
    
    # random fixed range
    if isinstance(msg, mido.MetaMessage) and msg.type == 'set_tempo':
        bpm, _ = randomParameter()
        msg.tempo = round(bpmToMidoTempo(bpm))
    bpm, change_tempo = randomParameter()
    if isinstance(msg, mido.Message) and msg.type == 'note_on' and msg.velocity>0 and msg.time>0 and change_tempo == 1:
        tempo = bpmToMidoTempo(bpm)
        new_msg = mido.MetaMessage('set_tempo', tempo=round(tempo), time=0)
        new_track.append(new_msg)
        print("new_msg: ", bpm, new_msg)
    new_track.append(msg)
    print("msg: ", vars(msg), isinstance(msg, mido.MetaMessage))
    
mo.tracks.append(new_track)

for m in new_track:
    print(m)

for t in mid.tracks[1:]:
    new_track = mido.MidiTrack()
    for msg in t:
        new_track.append(msg)
    mo.tracks.append(new_track)

mo.save('./assessment/new_song.mid')
