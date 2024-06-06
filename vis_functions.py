import os
import sys
from frame_processor import to_model_frames, phonetic_class_annotator
import librosa
import itertools
import pandas as pd
import textgrids


if "np" not in sys.modules:
    import numpy as np


# =========================================================================================
# Plot axis calculator
# =========================================================================================
# get the major and minor tick locations for the plot
def major_minor_calculator(phd, chosen_frames_, spf, round_up):
    """
    :param phd: phonetic detail
    :param chosen_frames_: start time and stop time [start, stop] for annotation
    :param spf: sample per wav2vec2 frame (len(audio) / len(wav2vec2_frames))
    :param round_up: True: round up, False: round nearest integer
    :return:
    """
    phd_start = to_model_frames(phd["start"], spf, round_up)
    phd_stop = to_model_frames(phd["stop"], spf, round_up)

    majors = {"ticks": [], "labels": []}
    minors = {"ticks": [], "labels": []}

    last_phone_indx = 0
    for j in range(len(phd['utterance'])):
        if phd_start[j] >= chosen_frames_[0] and phd_stop[j] <= chosen_frames_[1]:
            majors["labels"].append(phd['utterance'][j])
            majors["ticks"].append((phd_stop[j] + phd_start[j]) / 2)
            try:
                minors["labels"].append(f"{phd['utterance'][j - 1]},{phd['utterance'][j]}")
                minors["ticks"].append(phd_stop[j - 1])
            except:
                continue
            last_phone_indx = j

    try:
        minors["labels"].append(f"{phd['utterance'][last_phone_indx]},{phd['utterance'][last_phone_indx + 1]}")
        minors["ticks"].append(phd_stop[last_phone_indx])
    except:
        pass

    if chosen_frames_[1] == phd_stop[-1]:
        majors["labels"].append(phd['utterance'][-1])
        majors["ticks"].append((phd_stop[-1] + phd_start[-1]) // 2)

    # because of the way matplotlib ticker works it cannot handle overlaps and just adds lables to the next ticks with
    # no overlap!
    #
    overlaps = []
    for k in range(len(minors["ticks"]) - 1):
        if minors["ticks"][k] == minors["ticks"][k + 1]:
            print("similar", minors["labels"][k], minors["labels"][k + 1], minors["ticks"][k])
            print(",".join([minors["labels"][k], minors["labels"][k + 1].split(",")[1:][0]]), minors["ticks"][k])
            overlaps.append(k)
            # print(overlaps)
            minors["labels"][k] = ",".join([minors["labels"][k], minors["labels"][k + 1].split(",")[1:][0]])
            minors["labels"][k + 1] = ""

    fixed = -1
    for i, mj in enumerate(majors["ticks"]):
        for j, mn in enumerate(minors["ticks"]):
            if mn == mj and fixed != j - 1:
                print(j)
                majors["labels"][i] = f"{majors['labels'][i]} ({minors['labels'][j]})"
                minors['labels'][j] = ""
                fixed = j

    empty_minor = []
    for i, mn in enumerate(minors["labels"]):
        if mn == "":
            empty_minor.append(i - len(empty_minor))

    for i in empty_minor:
        # print("empty", mn, i, minors["ticks"][i])
        minors["labels"] = np.concatenate((minors["labels"][:i], minors["labels"][i + 1:]), axis=None)

    # print(len(minors["ticks"]), len(minors["labels"]))
    # print(len(majors["ticks"]), len(majors["labels"]))

    minors["ticks"] = np.array(minors["ticks"]) + 0.5
    majors["ticks"] = np.array(majors["ticks"]) + 0.5

    return majors, minors


def word_selector(data_, speaker_n, selection_, round_up, audio=None, update_=False, dummy_phones=False, debug=False):
    words = list(selection_)
    data_[speaker_n]["words"] = words
    max_minor_ticks = 0

    if debug:
        print(f"========================= {speaker_n} =========================")
        print(words)

    # if the function is not running inside the update function then get the timestamps to be stored in data
    if not update_:
        if debug:
            print(data_[speaker_n]["spf"])
        word_starts = to_model_frames(audio["word_detail"]["start"], data_[speaker_n]["spf"], round_up)
        data_[speaker_n]["word_starts"] = word_starts
        word_stops = to_model_frames(audio["word_detail"]["stop"], data_[speaker_n]["spf"], round_up)
        data_[speaker_n]["word_stops"] = word_stops
        phone_starts = to_model_frames(audio["phonetic_detail"]["start"], data_[speaker_n]["spf"], round_up)
        data_[speaker_n]["phone_starts"] = phone_starts
        phone_stops = to_model_frames(audio["phonetic_detail"]["stop"], data_[speaker_n]["spf"], round_up)
        data_[speaker_n]["phone_stops"] = phone_stops

    # chosen frames from start to end
    chosen_frames_w = [data_[speaker_n]["word_starts"][words[0]], data_[speaker_n]["word_stops"][words[-1]]]
    data_[speaker_n]["chosen_frames_w"] = chosen_frames_w

    # add the phones to the list
    chosen_phones_og = []
    chosen_phones_starts = []
    chosen_phones_stops = []
    for i, p in enumerate(data_[speaker_n]["all_phones"]):
        if data_[speaker_n]["phone_starts"][i] >= data_[speaker_n]["chosen_frames_w"][0] and \
                data_[speaker_n]["phone_stops"][i] <= data_[speaker_n]["chosen_frames_w"][1]:
            chosen_phones_og.append(p)
            chosen_phones_starts.append(data_[speaker_n]["phone_starts"][i])
            chosen_phones_stops.append(data_[speaker_n]["phone_stops"][i])

    data_[speaker_n]["chosen_phones_og"] = chosen_phones_og
    data_[speaker_n]["chosen_phones_starts"] = chosen_phones_starts
    data_[speaker_n]["chosen_phones_stops"] = chosen_phones_stops

    # create phone list with boundaries for frame based plot (including one timestamp before and after)
    phones = phonetic_class_annotator(data_[speaker_n]["phonetic_detail"], data_[speaker_n]["spf"], phone_only=True)["phone"][
             data_[speaker_n]["chosen_frames_w"][0] - 1:data_[speaker_n]["chosen_frames_w"][-1] + 2]

    # if there are identical start and stop for the last frame append the same frame at the end again to avoid
    # losing the frame
    if data_[speaker_n]["chosen_phones_starts"][-1] == data_[speaker_n]["chosen_phones_stops"][-1]:
        tmp_starts = data_[speaker_n]["chosen_phones_starts"]
    else:
        tmp_starts = np.append(data_[speaker_n]["chosen_phones_starts"], data_[speaker_n]["chosen_phones_stops"][-1])

    # create phone boundaries
    for start in tmp_starts:
        # print("STRT", start)
        str_ = phones.loc[start - 1]
        stp_ = phones.loc[start]
        if "," in str_:
            str_ = str_.split(",")[-1]
        if "," in stp_:
            stp_ = stp_.split(",")[-1]
        phones.loc[start] = f"{str_},{stp_}"

    phones = phones.drop(index=data_[speaker_n]["chosen_frames_w"][0] - 1)
    phones = phones.drop(index=data_[speaker_n]["chosen_frames_w"][1] + 1)
    chosen_phones = phones.values
    data_[speaker_n]["chosen_phones"] = chosen_phones

    word_list = data_[speaker_n]["all_words"][words[0]:words[-1] + 1]
    data_[speaker_n]["word_list"] = word_list

    if debug:
        print("\nword".ljust(12), "start".ljust(5), "stop", f"\n{'-'*24}")
        for i in words:
            print(data_[speaker_n]["all_words"][i].ljust(12), str(data_[speaker_n]["word_starts"][i]).ljust(5),
                  data_[speaker_n]["word_stops"][i])

        print("\nframes:", data_[speaker_n]["chosen_frames_w"])
        print("number of frames:",
              len(list(range(data_[speaker_n]["chosen_frames_w"][0], data_[speaker_n]["chosen_frames_w"][-1] + 1))))

        print("\nphone".ljust(12), "start".ljust(5), "stop", f"\n{'-'*24}")
        for i in range(len(chosen_phones_og)):
            print(chosen_phones_og[i].ljust(12), str(data_[speaker_n]["chosen_phones_starts"][i]).ljust(5),
                  data_[speaker_n]["chosen_phones_stops"][i])

        # print("CP start:", chosen_phones_starts)
        # print("CP stop:", chosen_phones_stops)
        # print("CPs:", chosen_phones)
        # print("WL:", word_list)

    majors_, minors_ = major_minor_calculator(data_[speaker_n]["phonetic_detail"],
                                              data_[speaker_n]["chosen_frames_w"],
                                              data_[speaker_n]["spf"], round_up)

    data_[speaker_n]["majors"] = majors_
    data_[speaker_n]["minors"] = minors_

    # for drawing lines at the minor ticks
    data_[speaker_n]["max_minor_ticks"] = max(max_minor_ticks, len(minors_["ticks"]))

    # end of word selector


def draw_lines(data_, ax_, speaker_):
    '''
    Draws lines for phone boudaries
    :param data_: utterance dictionary
    :param ax_: the plot axes
    :param speaker_: speaker name
    '''
    # print(data_[speaker_]["max_minor_ticks"])
    for i in range(data_[speaker_]["max_minor_ticks"] * 2):
        ax_.plot([data_[speaker_]["minors"]["ticks"][0]], [data_[speaker_]["minors"]["ticks"][0]],
                 color='black', linewidth=0.8, alpha=0.5)

    line_no = 0
    frame_count = data_[speaker_]["chosen_frames_w"][1] - data_[speaker_]["chosen_frames_w"][0] + 1
    for i in range(len(data_[speaker_]["minors"]["ticks"])):
        x = list(range(int(data_[speaker_]["minors"]["ticks"][0]), int(data_[speaker_]["minors"]["ticks"][-1]) + 2))
        y = [data_[speaker_]["minors"]["ticks"][i]] * (frame_count + 1)
        # Plot a phone boundary lines
        ax_.get_lines()[line_no].set_data(x, y)
        line_no += 1
        ax_.get_lines()[line_no].set_data(y, x)
        line_no += 1



# creat a dictionary for the audio file
# use the provided details for alignments or generate dummy word and phonetic annotation for unannotated files
def generate_audio_details(file_name, phonetic_detail, word_detail, sampling_rate=16000):
    utterance = {}
    speech, _ = librosa.load(file_name, sr=sampling_rate)
    audio_len_og = len(speech)
    print("og_audio_len:", audio_len_og)
    utterance["id"] = file_name.lower()
    utterance["audio"] = {}
    utterance["audio"]["sampling_rate"] = sampling_rate
    utterance["audio"]["path"] = file_name
    phn_time = 0.040  # ms time for each arbiterary phone section
    if word_detail is not None and phonetic_detail is not None:
        utterance["word_detail"] = word_detail
        utterance["phonetic_detail"] = phonetic_detail
    else:
        # pad speech with 0s becasue we don't know the exact silence period at the beginning.
        pad_len = 800
        speech = np.hstack((np.zeros(pad_len), speech, np.zeros(pad_len)))
        audio_len = len(speech)
        print("padded_audio_len:", audio_len)
        # Word detail (125 ms for each dummy word)
        word_dist = list((np.arange(start=pad_len, stop=len(speech), step=sampling_rate * (phn_time * 5))).astype(int))
        word_dist[-1] = audio_len_og + pad_len
        words = [f"t{i}" for i in range(len(word_dist)-1)]
        words_start = word_dist[0:-1]
        words_stop = word_dist[1:]
        utterance["word_detail"] = {}
        utterance["word_detail"]["utterance"] = words
        utterance["word_detail"]["start"] = words_start
        utterance["word_detail"]["stop"] = words_stop

        # Phonetic Detail
        utterance["phonetic_detail"] = {}
        # letters = ['b', 'd', 'f']
        # phon_iterator = itertools.cycle(letters)
        # add the padded silence to the begining
        utterance["phonetic_detail"]["utterance"] = ['h#']
        utterance["phonetic_detail"]["start"] = [0]
        utterance["phonetic_detail"]["stop"] = [pad_len]
        num_phones = 0
        for w, word in enumerate(utterance["word_detail"]["utterance"]):
            # 50 ms for each dummy phone
            phon_dist = list((np.arange(start=utterance["word_detail"]["start"][w],
                                        stop=utterance["word_detail"]["stop"][w], step=sampling_rate * phn_time)).astype(int))

            if phon_dist[-1] < utterance["word_detail"]["stop"][w]:
                phon_dist.append(utterance["word_detail"]["stop"][w])
            else:
                phon_dist[-1] =utterance["word_detail"]["stop"][w]

            for n in range(len(phon_dist)-1):
                utterance["phonetic_detail"]["utterance"].append(str(num_phones))
                utterance["phonetic_detail"]["start"].append(phon_dist[n])
                utterance["phonetic_detail"]["stop"].append(phon_dist[n+1])
                num_phones += 1

        # add the padded silence to the end
        utterance["phonetic_detail"]["utterance"].append("h#")
        utterance["phonetic_detail"]["start"].append(audio_len_og + pad_len)
        utterance["phonetic_detail"]["stop"].append(audio_len_og + (pad_len * 2))

    utterance["audio"]["array"] = speech

    return utterance


# use this if you have utterances in TIMIT style annotation
def timitdata_reader(wavfilepath, phn_ext=None, wrd_ext=None, txt_ext=None):
    # print(wavfilepath[:-4])
    speech_, _ = librosa.load(wavfilepath, sr=16000)
    txt_file = wavfilepath[:-3] + f"{'txt' if txt_ext is None else txt_ext}"
    phn_file = wavfilepath[:-3] + f"{'phn' if phn_ext is None else phn_ext}"
    wrd_file = wavfilepath[:-3] + f"{'wrd' if wrd_ext is None else wrd_ext}"

    # see if text transcription exists
    if os.path.isfile(txt_file):
        txt = pd.read_fwf(txt_file, header=None)
        text = ' '.join(list(txt.loc[0][2:]))
    else:
        print(f"Text details not found: ({txt_file})")
        text = ""

    # get the phonetic transcriptions if exists, generated dummy phone if not
    if os.path.isfile(phn_file):
        dummy_phones = False
        phn_detail = pd.read_csv(phn_file, header=None, names=['start', 'stop', 'utterance'],
                                 delim_whitespace=True)
    else:
        print(f"Phonetic details not found: ({phn_file})")
        dummy_phones = True
        tmp = generate_audio_details(wavfilepath, None, None, 16000)
        phn_detail = tmp["phonetic_detail"]


    # get the word transcriptions, or generate dummy words based on the phones
    if os.path.isfile(wrd_file):
        dummy_words = False
        wrd_detail = pd.read_csv(wrd_file, header=None, names=['start', 'stop', 'utterance'],
                                 delim_whitespace=True)
        # if the phones are dummies, need to re-adjust them to the word transcriptiosn
        if dummy_phones:
            phn_detail = wrd_detail.to_dict(orient="list")
            # print(phn_detail)

            phn_detail["utterance"] = ["h#"] + [f"{i}" for i in range(0, len(phn_detail["start"]))] + [f"h#"]
            phn_detail["start"] = [0] + list(phn_detail["start"]) + [phn_detail["stop"][len(phn_detail["stop"])-1]]
            phn_detail["stop"] = [phn_detail["start"][1]] + list(phn_detail["stop"]) + [len(speech_)]
            for phn_time in range(len(phn_detail["stop"])-1):
                if phn_detail["stop"][phn_time] > phn_detail["start"][phn_time+1]:
                    phn_detail["stop"][phn_time] = phn_detail["start"][phn_time+1]
                elif phn_detail["stop"][phn_time] < phn_detail["start"][phn_time+1]:
                    phn_detail["start"][phn_time+1] = phn_detail["stop"][phn_time]
            # print(phn_detail)
    else:
        print(f"Word details not found: ({wrd_file})")
        dummy_words = True
        tmp = generate_audio_details(wavfilepath, None, None, 16000)
        if dummy_phones:
            wrd_detail = tmp["word_detail"]
        else:
            wrd_detail = {"utterance":[f"ts{i}" for i in range(len(phn_detail["start"])//3)],
                           "start":[phn_detail["start"][i] for i in range(1, len(phn_detail["start"]), 3)],
                           "stop":[phn_detail["stop"][i] for i in range(3, len(phn_detail["start"]), 3)]}
            # pandas doesn't accept negative index values, need to use a compatible woraround
            if len(wrd_detail["utterance"]) > len(wrd_detail["start"]):
                print("dw>", len(wrd_detail["utterance"]) - len(wrd_detail["start"]))
                wrd_detail["start"].append(phn_detail['start'][len(phn_detail['start']) - 2])
                wrd_detail["stop"].append(phn_detail['stop'][len(phn_detail['stop']) - 2])
            elif len(wrd_detail["utterance"]) < len(wrd_detail["start"]):
                print("dw<", len(wrd_detail["utterance"]) - len(wrd_detail["start"]))
                wrd_detail["start"].pop(-1)
                wrd_detail["stop"].pop(-1)
            elif len(wrd_detail["utterance"]) == len(wrd_detail["start"]):
                print("dw=", len(wrd_detail["utterance"]) - len(wrd_detail["start"]))
                wrd_detail["start"][-1] = phn_detail['start'][len(phn_detail['start']) - 2]
                wrd_detail["stop"][-1] = phn_detail['stop'][len(phn_detail['stop']) - 2]
            # print("phn", phn_detail)
            # print("wrd", wrd_detail)


    sep_c = os.sep

    # a naive way to see if the TIMIT file is included in the original folder structure
    if wavfilepath.split(sep_c) in ["train", "test"]:
        sampl_ = {'file': wavfilepath,
                  'audio': {'path': wavfilepath, 'array': speech_, 'sampling_rate': 16000},
                  'text': text,
                  'phonetic_detail': {'start': list(phn_detail["start"]), 'stop': list(phn_detail["stop"]),
                                      'utterance': list(phn_detail["utterance"])},
                  'word_detail': {'start': list(wrd_detail["start"]), 'stop': list(wrd_detail["stop"]),
                                  'utterance': list(wrd_detail["utterance"])},
                  'dialect_region': wavfilepath.split(sep_c)[-3],
                  'sentence_type': wavfilepath.split(sep_c)[-1].split(".")[0][0:2],
                  'speaker_id': wavfilepath.split(sep_c)[-2],
                  'id': wavfilepath.split(sep_c)[-1].split(".")[0]}
    else:
        sampl_ = {'file': wavfilepath,
                  'audio': {'path': wavfilepath, 'array': speech_, 'sampling_rate': 16000},
                  'text': text,
                  'phonetic_detail': {'start': list(phn_detail["start"]), 'stop': list(phn_detail["stop"]),
                                      'utterance': list(phn_detail["utterance"])},
                  'word_detail': {'start': list(wrd_detail["start"]), 'stop': list(wrd_detail["stop"]),
                                  'utterance': list(wrd_detail["utterance"])},
                  'sentence_type': wavfilepath.split(sep_c)[-1].split(".")[0][0:2],
                  'id': wavfilepath.split(sep_c)[-1].split(".")[0]}

    return sampl_



def textgrid_reader(wavfilepath, phone_tier=None, word_tier=None, text_tier=None, tg_ext="textgrid", sample_rate=16000):
    # print(wavfilepath[:-4])
    try:
        grid = textgrids.TextGrid(wavfilepath[:-4]+'.'+tg_ext)
    except Exception as e:
        raise ValueError(f"{e}")

    txt_file = wavfilepath[:-4] + '.txt'
    if text_tier in grid:
        text = grid[text_tier].text.transcode()
    elif os.path.isfile(txt_file):
        txt = pd.read_fwf(txt_file, header=None)
        text = ' '.join(list(txt.loc[0][:]))
    elif text_tier is None:
        text = ""
    else:
        raise ValueError(f"Invalid text tier name: {text_tier}.")

    speech_, _ = librosa.load(wavfilepath, sr=sample_rate)


    # get the phonetic transcriptions if exists, generated dummy phone if not
    if phone_tier is not None and phone_tier in grid:
        dummy_phones = False
        phn_detail = {"start": [], "stop":[], "utterance":[]}
        phn_detail["utterance"] = [grid[phone_tier][phone].text for phone in range(len(grid[phone_tier]))]
        phn_detail["start"] = [grid[phone_tier][phone].xmin * sample_rate for phone in range(len(grid[phone_tier]))]
        phn_detail["stop"] = [grid[phone_tier][phone].xmax * sample_rate for phone in range(len(grid[phone_tier]))]
    else:
        print("Phonetic details not found.")
        dummy_phones = True
        tmp = generate_audio_details(wavfilepath, None, None, 16000)
        phn_detail = tmp["phonetic_detail"]


    # get the word transcriptions, or generate dummy words based on the phones
    if word_tier is not None and word_tier in grid:
        dummy_words = False
        wrd_detail = {"start": [], "stop":[], "utterance":[]}
        wrd_detail["utterance"] = [grid[word_tier][phone].text for phone in range(len(grid[word_tier]))]
        wrd_detail["start"] = [grid[word_tier][phone].xmin * sample_rate for phone in range(len(grid[word_tier]))]
        wrd_detail["stop"] = [grid[word_tier][phone].xmax * sample_rate for phone in range(len(grid[word_tier]))]
        # remove silences from start and end of words
        for ind in [0, -1]:
            if wrd_detail["utterance"][ind] == "":
                wrd_detail["utterance"].pop(ind)
                wrd_detail["start"].pop(ind)
                wrd_detail["stop"].pop(ind)
        # if the phones are dummies, need to re-adjust them to the word transcriptiosn
        if dummy_phones:
            phn_detail = wrd_detail.copy()
            # print("phn from wrd", phn_detail)

            phn_detail["utterance"] = ["h#"] + [f"{i}" for i in range(0, len(phn_detail["start"]))] + [f"h#"]
            phn_detail["start"] = [0] + list(phn_detail["start"]) + [phn_detail["stop"][len(phn_detail["stop"])-1]]
            phn_detail["stop"] = [phn_detail["start"][1]] + list(phn_detail["stop"]) + [len(speech_)]
            for phn_time in range(len(phn_detail["stop"])-1):
                if phn_detail["stop"][phn_time] > phn_detail["start"][phn_time+1]:
                    phn_detail["stop"][phn_time] = phn_detail["start"][phn_time+1]
                elif phn_detail["stop"][phn_time] < phn_detail["start"][phn_time+1]:
                    phn_detail["start"][phn_time+1] = phn_detail["stop"][phn_time]
            # print("final phn  ", phn_detail)
    else:
        print("Word details not found.")
        dummy_words = True
        tmp = generate_audio_details(wavfilepath, None, None, 16000)
        if dummy_phones:
            wrd_detail = tmp["word_detail"]
        else:
            for i in range(1, len(phn_detail), 2):
                print(i, phn_detail["start"][i])
            wrd_detail = {"utterance":[f"ts{i}" for i in range(len(phn_detail["start"])//3)],
                           "start":[phn_detail["start"][i] for i in range(1, len(phn_detail["start"]), 3)],
                           "stop":[phn_detail["stop"][i] for i in range(3, len(phn_detail["stop"]), 3)]}
            # pandas doesn't accept negative index values, need to use a compatible workround
            if len(wrd_detail["utterance"]) > len(wrd_detail["start"]):
                print("dw>", len(wrd_detail["utterance"]) - len(wrd_detail["start"]))
                wrd_detail["start"].append(phn_detail['start'][len(phn_detail['start']) - 2])
                wrd_detail["stop"].append(phn_detail['stop'][len(phn_detail['stop']) - 2])
            elif len(wrd_detail["utterance"]) < len(wrd_detail["start"]):
                print("dw<", len(wrd_detail["utterance"]) - len(wrd_detail["start"]))
                wrd_detail["start"].pop(-1)
                wrd_detail["stop"].pop(-1)
            elif len(wrd_detail["utterance"]) == len(wrd_detail["start"]):
                wrd_detail["start"][-1] = phn_detail['start'][len(phn_detail['start'])-2]
                wrd_detail["stop"][-1] = phn_detail['stop'][len(phn_detail['stop'])-2]
            # print("phn", phn_detail)
            # print("wrd", wrd_detail)


    sep_c = os.sep

    sampl_ = {'file': wavfilepath,
              'audio': {'path': wavfilepath, 'array': speech_, 'sampling_rate': 16000},
              'text': text,
              'phonetic_detail': {'start': list(phn_detail["start"]), 'stop': list(phn_detail["stop"]),
                                  'utterance': list(phn_detail["utterance"])},
              'word_detail': {'start': list(wrd_detail["start"]), 'stop': list(wrd_detail["stop"]),
                              'utterance': list(wrd_detail["utterance"])},
              'sentence_type': wavfilepath.split(sep_c)[-1].split(".")[0][0:2],
              'id': wavfilepath.split(sep_c)[-1].split(".")[0]}

    return sampl_


def alignment_reader(wavfilepath, alignments, phone_tier=None, word_tier=None, text_tier=None, tg_ext="textgrid",
                    sample_rate=16000):
    """
    WorkInProgress
    :param wavfilepath:
    :param alignments:
    :param phone_tier:
    :param word_tier:
    :param text_tier:
    :param tg_ext:
    :param sample_rate:
    :return:
    """
    # here, tiers can be file extension for timit style files.
    txt_file = wavfilepath[:-4] + text_tier
    phn_file = wavfilepath[:-4] + phone_tier
    wrd_file = wavfilepath[:-4] + word_tier

    if alignments == "textgrid":
        try:
            grid = textgrids.TextGrid(wavfilepath[:-4] + '.' + tg_ext)
        except Exception as e:
            raise ValueError(f"{e}")

    if alignments.lower() == "timit" and os.path.isfile(txt_file):
        txt = pd.read_fwf(txt_file, header=None)
        text = ' '.join(list(txt.loc[0][2:]))
    elif alignments.lower() == "textgrid" and text_tier in grid:
        text = grid[text_tier].text.transcode()
    else:
        text = ""

    speech_, _ = librosa.load(wavfilepath, sr=sample_rate)

    # get the phonetic transcriptions if exists, generated dummy phone if not
    if alignments.lower() == "timit" and os.path.isfile(phn_file):
        dummy_phones = False
        phn_detail = pd.read_csv(phn_file, header=None, names=['start', 'stop', 'utterance'],
                                 delim_whitespace=True)
    elif alignments.lower() == "textgrid" and phone_tier is not None and phone_tier in grid:
        dummy_phones = False
        phn_detail = {"start": [], "stop": [], "utterance": []}
        phn_detail["utterance"] = [grid[phone_tier][phone].text for phone in range(len(grid[phone_tier]))]
        phn_detail["start"] = [grid[phone_tier][phone].xmin * sample_rate for phone in range(len(grid[phone_tier]))]
        phn_detail["stop"] = [grid[phone_tier][phone].xmax * sample_rate for phone in range(len(grid[phone_tier]))]
    else:
        print("Phonetic details not found.")
        dummy_phones = True
        tmp = generate_audio_details(wavfilepath, None, None, 16000)
        phn_detail = tmp["phonetic_detail"]


    # get the word transcriptions, or generate dummy words based on the phones
    if alignments.lower() == "timit" and os.path.isfile(wrd_file):
        dummy_words = False
        wrd_detail = pd.read_csv(wrd_file, header=None, names=['start', 'stop', 'utterance'],
                                 delim_whitespace=True)
        # if the phones are dummies, need to re-adjust them to the word transcriptiosn
        if dummy_phones:
            phn_detail = wrd_detail.to_dict(orient="list")
            # print(phn_detail)

            phn_detail["utterance"] = ["h#"] + [f"{i}" for i in range(0, len(phn_detail["start"]))] + [f"h#"]
            phn_detail["start"] = [0] + list(phn_detail["start"]) + [phn_detail["stop"][len(phn_detail["stop"]) - 1]]
            phn_detail["stop"] = [phn_detail["start"][1]] + list(phn_detail["stop"]) + [len(speech_)]
            for phn_time in range(len(phn_detail["stop"]) - 1):
                if phn_detail["stop"][phn_time] > phn_detail["start"][phn_time + 1]:
                    phn_detail["stop"][phn_time] = phn_detail["start"][phn_time + 1]
                elif phn_detail["stop"][phn_time] < phn_detail["start"][phn_time + 1]:
                    phn_detail["start"][phn_time + 1] = phn_detail["stop"][phn_time]
    elif alignments.lower() == "textgrid" and word_tier is not None and word_tier in grid:
        dummy_words = False
        wrd_detail = {"start": [], "stop": [], "utterance": []}
        wrd_detail["utterance"] = [grid[word_tier][phone].text for phone in range(len(grid[word_tier]))]
        wrd_detail["start"] = [grid[word_tier][phone].xmin * sample_rate for phone in range(len(grid[word_tier]))]
        wrd_detail["stop"] = [grid[word_tier][phone].xmax * sample_rate for phone in range(len(grid[word_tier]))]
        # remove silences from start and end of words
        for ind in [0, -1]:
            if wrd_detail["utterance"][ind] == "":
                wrd_detail["utterance"].pop(ind)
                wrd_detail["start"].pop(ind)
                wrd_detail["stop"].pop(ind)
        print(wrd_detail)
        # if the phones are dummies, need to re-adjust them to the word transcriptiosn
        if dummy_phones:
            phn_detail = wrd_detail.copy()
            # print("phn from wrd", phn_detail)

            phn_detail["utterance"] = ["h#"] + [f"{i}" for i in range(0, len(phn_detail["start"]))] + [f"h#"]
            phn_detail["start"] = [0] + list(phn_detail["start"]) + [
                phn_detail["stop"][len(phn_detail["stop"]) - 1]]
            phn_detail["stop"] = [phn_detail["start"][1]] + list(phn_detail["stop"]) + [len(speech_)]
            for phn_time in range(len(phn_detail["stop"]) - 1):
                if phn_detail["stop"][phn_time] > phn_detail["start"][phn_time + 1]:
                    phn_detail["stop"][phn_time] = phn_detail["start"][phn_time + 1]
                elif phn_detail["stop"][phn_time] < phn_detail["start"][phn_time + 1]:
                    phn_detail["start"][phn_time + 1] = phn_detail["stop"][phn_time]
            # print("final phn  ", phn_detail)
    else:
        print("Word details not found.")
        dummy_words = True
        tmp = generate_audio_details(wavfilepath, None, None, 16000)
        if dummy_phones:
            wrd_detail = tmp["word_detail"]
        else:
            wrd_detail = {"utterance": [f"ts{i}" for i in range(len(phn_detail["start"]) // 3)],
                          "start": [phn_detail["start"][i] for i in range(1, len(phn_detail["start"]), 3)],
                          "stop": [phn_detail["stop"][i] for i in range(1, len(phn_detail["start"]), 3)]}
            # print(wrd_detail)
            # pandas doesn't accept negative index values, need to use a compatible woraround
            if len(wrd_detail["utterance"]) > len(wrd_detail["start"]):
                wrd_detail["start"].append(phn_detail['start'][len(phn_detail['start']) - 2])
                wrd_detail["stop"].append(phn_detail['stop'][len(phn_detail['start']) - 2])
            if len(wrd_detail["utterance"]) < len(wrd_detail["start"]):
                wrd_detail["start"].pop(-1)
                wrd_detail["stop"].pop(-1)
            elif len(wrd_detail["utterance"]) == len(wrd_detail["start"]):
                wrd_detail["start"][-1] = phn_detail['start'][len(phn_detail['start']) - 2]
                wrd_detail["stop"][-1] = phn_detail['stop'][len(phn_detail['start']) - 2]
            # print("phn", phn_detail)
            # print("wrd", wrd_detail)

    sep_c = os.sep

    sampl_ = {'file': wavfilepath,
              'audio': {'path': wavfilepath, 'array': speech_, 'sampling_rate': 16000},
              'text': text,
              'phonetic_detail': {'start': list(phn_detail["start"]), 'stop': list(phn_detail["stop"]),
                                  'utterance': list(phn_detail["utterance"])},
              'word_detail': {'start': list(wrd_detail["start"]), 'stop': list(wrd_detail["stop"]),
                              'utterance': list(wrd_detail["utterance"])},
              'sentence_type': wavfilepath.split(sep_c)[-1].split(".")[0][0:2],
              'id': wavfilepath.split(sep_c)[-1].split(".")[0]}

    return sampl_