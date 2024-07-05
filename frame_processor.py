import torch
import numpy as np
import pandas as pd
import torch
from statistics import median

from matplotlib import pyplot as plt, ticker
from scipy.signal import find_peaks


# Parameters:
# details: TIMIT phonetic details
def to_model_frames(timestamps, spf, round_up=False):
    # for sample-based timestamps: spf = samples per frame = len(audio array) / len(w2v2 representation frames)
    # For whisper it is always 320
    if round_up:
        return np.ceil(np.array(timestamps) / spf).astype(int)
    else:
        return (np.array(timestamps) / spf).round().astype(int)


def phonetic_class_annotator(phone_details, spf, mode="", debug=False, hidden_state_=None, p61=False, phone_only=False):

    if p61:
        map_table = pd.read_csv("resources/phonemapsat_60_49_40.csv")
        targets = pd.DataFrame(
            columns=['phone61', 'phone49', 'phone40', 'cat', 'poa61', 'moa61', 'voicing61', 'poa49', 'moa49',
                     'voicing49', 'poa40', 'moa40', 'voicing40', 'back61', 'height61', 'rounding61', 'back49',
                     'height49', 'rounding49', 'back40', 'height40', 'rounding40'])
        phone_c = "phone61"
    else:
        map_table = pd.read_csv("resources/phonemapsat.csv")
        targets = pd.DataFrame(columns=['phone', 'cat', 'poa', 'moa', 'voicing', 'height', 'back', 'rounding'])
        phone_c = "phone"

    start_ = to_model_frames(phone_details["start"], spf)
    stop_ = to_model_frames(phone_details["stop"], spf)

    frames_ = []
    # targets = []
    # print(phone_details["utterance"])
    # print(len(phone_details["utterance"]))
    if mode == "average":
        for indx, p in enumerate(phone_details["utterance"]):
            targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]
            frame_tmp = hidden_state_[start_[indx]:stop_[indx]+1]
            avg_ = np.average(frame_tmp, axis=0)  # .values
            # print(type(avg_))
            frames_.append(avg_)
    elif mode == "softmax":
        for indx, p in enumerate(phone_details["utterance"]):
            targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]
            sum_ = torch.sum(hidden_state_[start_[indx]:stop_[indx]+1], dim=0)
            softmax = torch.nn.Softmax(dim=0)(sum_).numpy()
            # print(type(softmax))
            frames_.append(softmax)
    elif mode == "max":
        for indx, p in enumerate(phone_details["utterance"]):
            targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]
            max_ = torch.max(hidden_state_[start_[indx]:stop_[indx]+1], dim=0).values.numpy()
            # print(type(max_))
            frames_.append(max_)
    else:
        overlap = False
        for indx, p in enumerate(phone_details["utterance"]):
            # print(indx)
            if debug:
                print(p)
            # if the previous frame was overlapping skip the first frame of the next timestamp
            if overlap:
                start_i = start_[indx] + 1
            else:
                start_i = start_[indx]

            if start_i == stop_[indx]:
                # if debug: print("equal start, stop:", start_i, p)
                if phone_only:
                    targets.loc[len(targets)] = p
                else:
                    targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]

                if hidden_state_ is not None:
                    # print(hidden_state_.shape)
                    frames_.append(hidden_state_[start_i])
                overlap = True
            elif start_i > stop_[indx]:
                if debug: print(f"Overlapping: {phone_details['utterance'][indx-1]}-{p}")
                overlap = True
            else:
                for t in range(start_i, stop_[indx]):
                    if phone_only:
                        targets.loc[len(targets)] = p
                    else:
                        targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == p].values[0]

                    if hidden_state_ is not None:
                        # print(hidden_state_.shape)
                        # frame_tmp.append(hidden_state_[t])
                        # frame_tmp = np.average(frame_tmp, axis=0)
                        # print(frame_tmp)
                        frames_.append(hidden_state_[t])
                overlap = False

        # if start_[-1] == stop_[-1]:
        #     # add the last frame
        #     if phone_only:
        #         targets.loc[len(targets)] = phone_details["utterance"][-1]
        #     else:
        #         targets.loc[len(targets)] = map_table.loc[map_table[phone_c] == phone_details["utterance"][-1]].values[0]
        #     if hidden_state_ is not None:
        #         frames_.append(hidden_state_[t+1])
        #     # print(p, targets[-1])

    if hidden_state_ is None:
        return targets
    else:
        return targets, frames_

