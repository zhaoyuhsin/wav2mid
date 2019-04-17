import soundfile
import torch
import numpy as np
SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048
audio_path="a1.wav"
tsv_path="a1.tsv"
data, samplerate = soundfile.read('test.wav')
print(samplerate)
print(len(data[0]))
audio, sr = soundfile.read("a1.wav", dtype='int16')
audio = torch.ShortTensor(audio)
audio_length = len(audio)
n_keys = MAX_MIDI - MIN_MIDI + 1
n_steps = (audio_length - 1) // HOP_LENGTH + 1
label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
print(label.size())
tsv_path = tsv_path
midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
print(len(midi))
for onset, offset, note, vel in midi:
    left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
    onset_right = min(n_steps, left + HOPS_IN_ONSET)
    frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
    frame_right = min(n_steps, frame_right)
    offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
    f = int(note) - MIN_MIDI
    label[left:onset_right, f] = 3
    label[onset_right:frame_right, f] = 2
    label[frame_right:offset_right, f] = 1
    velocity[left:frame_right, f] = vel
data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
print(label[0])
print(velocity[0])
