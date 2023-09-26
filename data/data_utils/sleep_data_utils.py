import os
import numpy as np
import pandas as pd
import h5py
import pickle
from scipy.signal import resample
from scipy import signal
import librosa

def read_dreem_data(data_dir, file_name):
    with h5py.File(os.path.join(data_dir, file_name), "r") as f:
        # labels = f["hypnogram"][()]
        ex_signals = ["FP1_F3",
                      "FP1_M2",
                      "FP1_O1",
                      "FP2_F4",
                      "FP2_M1",
                      "FP2_O2"]
        signals = []
        channels = []
        for key in f["signals"].keys():
            for ch in f["signals"][key].keys():
                if ch not in ex_signals:
                    signals.append(f["signals"][key][ch][()])
                    channels.append(ch)

    signals = np.stack(signals, axis=0)

    return signals, channels, data_dir
    # return signals, channels, labels

# scipy resampling
def read_dreem_data_pkl(data_dir, file_name, seq_len, to_freq):
    with open(os.path.join(data_dir, file_name), 'rb') as f:
        a = pickle.load(f)

        signals = []
        channels = []
        for key,value in a.items():
            if key=='Plethysmogram' or key=='Resp Rate':
                continue
            else:
                ori_freq = int(len(value) / seq_len)
                value = value[:seq_len*ori_freq]
                x = signal.resample(value, int(len(value) / (ori_freq/to_freq)),axis=0)
                signals.append(x)
                channels.append(key)
    signals = np.stack(signals, axis=0)

    return signals, channels, data_dir

# librosa resampling
# def read_dreem_data_pkl(data_dir, file_name, seq_len, to_freq):
#     with open(os.path.join(data_dir, file_name), 'rb') as f:
#         a = pickle.load(f)

#         signals = []
#         channels = []
#         for key,value in a.items():
#             if key!='Plethysmogram' and key!='EKG':
#                 value = np.array(value)
#                 # Get channel's original freq
#                 ori_freq = int(len(value)/seq_len)
#                 if ori_freq!=to_freq:
#                     # Resample ori_freq -> to_freq
#                     value = librosa.resample(value, orig_sr=ori_freq, target_sr=to_freq)
#                 signals.append(value)
#                 channels.append(key)
#     signals = np.stack(signals, axis=0)

#     return signals, channels, data_dir