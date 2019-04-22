import os
import soundfile
from onsets_and_frames import *
import numpy as np
from mir_eval.util import midi_to_hz

def getdata(audio_path):
    audio, sr = soundfile.read(audio_path, dtype='int16')
    assert sr == SAMPLE_RATE

    audio = torch.ShortTensor(audio)
    audio_length = len(audio)

    n_keys = MAX_MIDI - MIN_MIDI + 1
    n_steps = (audio_length - 1) // HOP_LENGTH + 1

    label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
    velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

    data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
    result = data
    result['audio'] = result['audio'].float().div_(32768.0)
    result['onset'] = (result['label'] == 3).float()
    result['offset'] = (result['label'] == 1).float()
    result['frame'] = (result['label'] > 1).float()
    result['velocity'] = result['velocity'].float().div_(128.0)
    return result

def eva(label, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    pred, losses = model.run_on_batch(label)

    for key, value in pred.items():
        value.squeeze_(0).relu_()

    p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], onset_threshold,
                                        frame_threshold)

    t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)
    return midi_path
def transfer(audio_path, save_path):
    format = audio_path[-3:]
    if (format == 'mp3'):
        filename=audio_path[:-3]
        cmd = "ffmpeg -y -loglevel fatal -i %smp3 -ac 1 -ar 16000 %sflac" % (filename, filename)
        print(cmd)
        os.system(cmd)
        audio_path=filename+"flac"
        print(audio_path)
    if (format == 'wav'):
        filename=audio_path[:-3]
        cmd = "ffmpeg -y -loglevel fatal -i %swav -ac 1 -ar 16000 %sflac" % (filename, filename)
        print(cmd)
        os.system(cmd)
        audio_path=filename+"flac"
        print(audio_path)
    data = getdata(audio_path)
    device='cpu'
    model = torch.load("model.pt", map_location=device).eval()
    return(eva(data, model,save_path=save_path))
transfer("blues.mp3", "output/")
