from vis_functions import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, Button, SpanSelector, RangeSlider, CheckButtons
import torch


class STEVE:

    def __init__(self):
        self.file_name = ""
        self.data = {}
        self.num_layers = 0
        self.num_attentions = 0
        self.num_speakers = 0
        self.round_up = False  # round up frame timestamp conversion
        self.dummy_phones = True


    def set_data(self, utterance, spkr, debug):
        '''
        :param utterance: a dictionary of speech file metadata generated using get_utterance()
        :param spkr: speaker name
        '''
        self.data[spkr]["all_words"] = utterance["word_detail"]["utterance"]
        self.data[spkr]["all_phones"] = utterance["phonetic_detail"]["utterance"]
        self.data[spkr]["phonetic_detail"] = utterance["phonetic_detail"]
        word_selector(self.data, spkr, range(0, 1), round_up=self.round_up, audio=utterance, update_=False,
                      dummy_phones=self.dummy_phones, debug=debug)
        self.frame_based_ = False


    def get_utterance(self, file_name, alignments, tg_phone, tg_word, tg_text, tg_ext):
        '''
        :param file_name: wav file name
        :param alignments: type of alignments: "timit" or "textgrid" or anything else for dummy alignments
        :param tg_phone: textgrid tier for phonetic alignment
        :param tg_word: textgrid tier for word alignment
        :param tg_text: textgrid tier for text alignment
        :param tg_ext: textgrid file extention
        :return: utterance dictionary with all the required meta data
        '''
        if alignments == "timit":
            utterance = timitdata_reader(file_name, tg_phone, tg_word, tg_text)
            # utterance = alignment_reader(file_name, alignments, sample_rate=16000)
            if ["b", "d", "f"] not in utterance["phonetic_detail"]["utterance"]:
                self.dummy_phones = False
        elif alignments == "textgrid":
            utterance = textgrid_reader(file_name, phone_tier=tg_phone, word_tier=tg_word, text_tier=tg_text,
                                        tg_ext=tg_ext, sample_rate=16000)
            # WIP: unify readers
            # utterance = alignment_reader(file_name, alignments, phone_tier=tg_phone, word_tier=tg_word, text_tier=tg_text,
            #                             tg_ext=tg_ext, sample_rate=16000)
            if ["b", "d", "f"] not in utterance["phonetic_detail"]["utterance"]:
                self.dummy_phones = False
        else:
            print(f"{bcolors.WARNING}Invalid alignment given ({alignments}), "
                  f"using dummy phones and words. Available alignments: \"timit\", \"textgrid\".{bcolors.ENDC}")
            self.dummy_phones = True
            phonetic_detail = None
            word_detail = None
            utterance = generate_audio_details(file_name, phonetic_detail, word_detail)

        return utterance

    def plot_attentions(self, file_name, attention, spf,
                        alignments=None, tg_phone=None, tg_word=None, tg_text=None, tg_ext="textgrid",
                        speakers=[], debug=False):
        '''
        Plots attention
        :param file_name: a single file path or a list of file paths
        :param attention: a single utterance self-attentions or multiples (order and number must match the file_name above)
        :param spf: audio sample per frame for a single utterance or a list of samples per utterance
        :param alignments: (optional) alignment types: "timit" or "textgrid" or anything else for dummy alignments
        :param tg_phone: (optional) textgrid tier for phonetic alignment
        :param tg_word: (optional) textgrid tier for word alignment
        :param tg_text: (optional) textgrid tier for the transcription of the utterance
        :param tg_ext: (optional) textgrid file extention, default is "textgrid"
        :param speakers: (optional) a list of speaker names for displaying in the visualiser
        :param debug: (optional) display debugging information
        '''

        # prepare utterances based on single file or multiple files
        if type(file_name) in [list, tuple, np.ndarray]:
            utterances = []
            for file in file_name:
                utterances.append(self.get_utterance(file, alignments, tg_phone, tg_word, tg_text, tg_ext))
        else:
            utterances = [self.get_utterance(file_name, alignments, tg_phone, tg_word, tg_text, tg_ext)]
            attention = [attention]
            spf = [spf]

        # see if nuber of utterance and attentions match
        if len(attention[0][0]) > 1 and len(utterances) > 1:
            raise Exception("You provided multiple utterances but only one model attentions.")

        if len(attention[0][0]) == 1:
            if len(attention) != len(utterances):
                raise Exception("Number of utterances does not match the number of model attentions.")

        # prepare data for plotting
        for u, utterance in enumerate(utterances):
            if speakers == []:
                sp = utterance["id"].split("/")[-1]
                sp = sp.split(os.sep)[-1]
                # sp = f"Utterance{u+1}"
            else:
                sp = speakers[u]

            self.num_layers = len(attention[u])
            self.num_attentions = len(attention[u][0][0])
            self.num_speakers += 1
            self.data[sp] = {}
            # Data keys:
            # ['tokens', 'spf', 'encoder_attn', 'all_words', 'all_phones', 'phonetic_detail', 'words', 'word_starts',
            # 'word_stops', 'phone_starts', 'phone_stops', 'chosen_frames_w', 'chosen_phones_og', 'chosen_phones_starts',
            # 'chosen_phones_stops', 'chosen_phones', 'word_list', 'majors', 'minors', 'max_minor_ticks']
            self.data[sp]["encoder_attn"] = attention[u]
            self.data[sp]["spf"]= spf[u]
            self.set_data(utterance, sp, debug)

            if self.data[sp]["all_words"][0] == "ts0":
                words_label = "Timestamp(s):"
            else:
                words_label = "Word(s):"

        # =========================================================================================
        # Initial Plot:
        # =========================================================================================
        speaker_ = list(self.data)[0]
        layer_num = 0
        head_num = 0

        frame_count = self.data[speaker_]["chosen_frames_w"][1] - self.data[speaker_]["chosen_frames_w"][0] + 1

        if debug:
            print(f"words: {' '.join(self.data[speaker_]['word_list'])}")
            print(f"phones: {self.data[speaker_]['chosen_phones']}")
            print(f"frames: {frame_count}")
            print()
            print(f"Encoder Attention Layer {layer_num}")
            print(f"Encoder Attention Head {head_num}")
        # plt.figure(figsize=(10, 10))
        self_attn = self.data[speaker_]["encoder_attn"][layer_num][0][head_num].cpu()
        self_attn = self_attn[self.data[speaker_]["chosen_frames_w"][0]:self.data[speaker_]["chosen_frames_w"][1] + 1,
                    self.data[speaker_]["chosen_frames_w"][0]:self.data[speaker_]["chosen_frames_w"][1] + 1]

        # ax = plt.gca()
        mpl.rcParams.update(mpl.rcParamsDefault)

        fig, ax = plt.subplots(figsize=(10, 9.5), num='Self-Attention Visualiser')

        plt.subplots_adjust(left=0.125, bottom=0.155, right=0.9, top=0.765, wspace=0, hspace=0)
        labelsize_base = 14
        ax.tick_params('both', which='major', labelrotation=0, length=50, pad=3, width=0.5, labelsize=labelsize_base)
        ax.tick_params(axis='x', which='major', labelrotation=90)
        ax.tick_params(axis='x', which='minor', labelrotation=90, length=5, pad=3, width=1, labelsize=labelsize_base-2)
        ax.tick_params(axis='y', which='minor', labelrotation=0, length=5, pad=3, width=1, labelsize=labelsize_base-2)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")

        # Y AXIS
        ax.yaxis.set_major_locator(ticker.FixedLocator(self.data[speaker_]["majors"]["ticks"]))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(self.data[speaker_]["majors"]["labels"]))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(self.data[speaker_]["minors"]["ticks"]))
        ax.yaxis.set_minor_formatter(ticker.FixedFormatter(self.data[speaker_]["minors"]["labels"]))

        # X AXIS
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_major_locator(ticker.FixedLocator(self.data[speaker_]["majors"]["ticks"]))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(self.data[speaker_]["majors"]["labels"]))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(self.data[speaker_]["minors"]["ticks"]))
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(self.data[speaker_]["minors"]["labels"]))

        # plt.xlabel("Phones/Words")
        # plt.ylabel("Phones/Words")

        interp_methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                          'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                          'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

        interp = 'bessel'
        # interp = 'none'

        im = ax.imshow(self_attn,
                       extent=[self.data[speaker_]["chosen_phones_starts"][0],
                               self.data[speaker_]["chosen_phones_stops"][-1] + 1,
                               self.data[speaker_]["chosen_phones_stops"][-1] + 1,
                               self.data[speaker_]["chosen_phones_starts"][0]],
                       interpolation=interp,
                       )

        # create maximum number of lines possible for the current utterance
        if not self.frame_based_:
            draw_lines(self.data, ax, speaker_)

        ax.title.set_text(f"{words_label} \'{' '.join(self.data[speaker_]['word_list'])}\' \nUtterance: {speaker_} | "
                          f"Encoder Attention Layer {layer_num}, Head {head_num}")

        # Make a horizontal sliders.
        # Speaker slider

        allowed_speaker = np.array([x for x in range(1, len(self.data) + 1)])
        axhauteur_speaker = plt.axes([0.2, 0.09, 0.65, 0.03])
        shauteur_speaker = Slider(axhauteur_speaker, 'Utterance', 1, len(self.data), valinit=0, valstep=allowed_speaker)
        shauteur_speaker.label.set_size(12)
        # Words slider
        allowed_words = np.array([x for x in range(0, len(self.data[speaker_]["all_words"]) + 1)])
        axhauteur_words = plt.axes([0.2, 0.065, 0.65, 0.03])
        shauteur_words = RangeSlider(axhauteur_words, 'Words', 0, len(self.data[speaker_]["all_words"]), valinit=(0, 1),
                                     valstep=allowed_words)
        shauteur_words.label.set_size(12)
        # Layer slider
        allowed_layer = np.array([x for x in range(0, self.num_layers)])
        axhauteur_layer = plt.axes([0.2, 0.04, 0.65, 0.03])
        shauteur_layer = Slider(axhauteur_layer, 'Layer', 0, self.num_layers - 1, valinit=0,
                                valstep=allowed_layer)
        shauteur_layer.label.set_size(12)
        # Head slider
        allowed_head = np.array([x for x in range(0, self.num_attentions)])
        axhauteur_head = plt.axes([0.2, 0.015, 0.65, 0.03])
        shauteur_head = Slider(axhauteur_head, 'Head', 0, self.num_attentions - 1, valinit=0,
                               valstep=allowed_head)
        shauteur_head.label.set_size(12)

        # Softmax Checkbox
        chkax = plt.axes([0.45, 0.12, 0.15, 0.03])
        chkax.spines['top'].set_visible(False)
        chkax.spines['right'].set_visible(False)
        chkax.spines['bottom'].set_visible(False)
        chkax.spines['left'].set_visible(False)
        check = CheckButtons(
            ax=chkax,
            labels=["Softmax"],
            actives=[False],
            label_props={'color': ['black']},
            frame_props={'edgecolor': ['black'], 'sizes': [(plt.rcParams['font.size'] / 1.2) ** 2]},
            check_props={'facecolor': ['black'], 'sizes': [(plt.rcParams['font.size'] / 1.2) ** 2]},
        )

        # The function to be called anytime a slider's value changes
        def update(val):
            speaker_ = list(self.data)[int(shauteur_speaker.val) - 1]
            word_list = shauteur_words.val
            layer_num = shauteur_layer.val
            head_num = shauteur_head.val
            softmax_ = check.get_status()[0]

            word_selector(self.data, speaker_, range(int(word_list[0]), int(word_list[1])), round_up=self.round_up,
                          update_=True, dummy_phones=self.dummy_phones, debug=debug)

            frame_count = self.data[speaker_]["chosen_frames_w"][1] - self.data[speaker_]["chosen_frames_w"][0] + 1

            self_attn = self.data[speaker_]["encoder_attn"][layer_num][0][head_num].cpu()

            if softmax_:
                self_attn = torch.nn.functional.softmax(self_attn, dim=-1)

            self_attn = self_attn[self.data[speaker_]["chosen_frames_w"][0]:self.data[speaker_]["chosen_frames_w"][1] + 1,
                        self.data[speaker_]["chosen_frames_w"][0]:self.data[speaker_]["chosen_frames_w"][1] + 1]
            self_attn = self_attn.numpy()


            if word_list[1] - word_list[0] > 3:
                ax.tick_params('both', which='major',
                               labelsize=labelsize_base - (0.625 * (word_list[1] - word_list[0])))
                ax.tick_params('both', which='minor',
                               labelsize=(labelsize_base - 2) - (0.625 * (word_list[1] - word_list[0])))
            else:
                ax.tick_params('both', which='major', labelsize=labelsize_base)
                ax.tick_params('both', which='minor', labelsize=labelsize_base - 2)

            # Y AXIS
            ax.yaxis.set_major_locator(ticker.FixedLocator(self.data[speaker_]["majors"]["ticks"]))
            ax.yaxis.set_major_formatter(ticker.FixedFormatter(self.data[speaker_]["majors"]["labels"]))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(self.data[speaker_]["minors"]["ticks"]))
            ax.yaxis.set_minor_formatter(ticker.FixedFormatter(self.data[speaker_]["minors"]["labels"]))

            # X AXIS
            ax.xaxis.set_label_position('top')
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_major_locator(ticker.FixedLocator(self.data[speaker_]["majors"]["ticks"]))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(self.data[speaker_]["majors"]["labels"]))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(self.data[speaker_]["minors"]["ticks"]))
            ax.xaxis.set_minor_formatter(ticker.FixedFormatter(self.data[speaker_]["minors"]["labels"]))

            if not self.frame_based_:
                # update the minors (phone border) lines
                total_ticks = len(self.data[speaker_]["minors"]["ticks"])
                total_lines = len(ax.get_lines()) // 2
                # print(f"total lines/ticks: {total_lines}, {total_ticks}")

                line_no = 0
                if total_lines < total_ticks:
                    for i in range((total_ticks - total_lines) * 2):
                        ax.plot([self.data[speaker_]["minors"]["ticks"][0]], [self.data[speaker_]["minors"]["ticks"][0]],
                                color='black', linewidth=0.8, alpha=0.5)

                total_ticks = len(self.data[speaker_]["minors"]["ticks"])
                total_lines = len(ax.get_lines()) // 2

                for i in range(total_ticks):
                    x = list(range(int(self.data[speaker_]["minors"]["ticks"][0]),
                                   int(self.data[speaker_]["minors"]["ticks"][-1]) + 2))
                    y = [self.data[speaker_]["minors"]["ticks"][i]] * (frame_count + 1)
                    ax.get_lines()[line_no].set_data(x, y)
                    line_no += 1
                    ax.get_lines()[line_no].set_data(y, x)
                    line_no += 1
                # get rid of extra lines
                if total_lines > total_ticks:
                    for i in range(line_no, (total_lines * 2)):
                        ax.get_lines()[i].set_data([self.data[speaker_]["minors"]["ticks"][0]],
                                                   [self.data[speaker_]["minors"]["ticks"][0]])

            im.set_data(self_attn)
            im.set_clim(vmin=np.min(self_attn), vmax=np.max(self_attn))
            im.set_extent([self.data[speaker_]["chosen_phones_starts"][0], self.data[speaker_]["chosen_phones_stops"][-1] + 1,
                           self.data[speaker_]["chosen_phones_stops"][-1] + 1,
                           self.data[speaker_]["chosen_phones_starts"][0]])

            ax.title.set_text(
                f"{words_label} \'{' '.join(self.data[speaker_]['all_words'][word_list[0]:word_list[1]])}\' \nUtterance: {speaker_} | "
                f"Encoder Attention Layer {layer_num}, Head {head_num}")

            fig.canvas.draw_idle()

        shauteur_speaker.on_changed(update)
        shauteur_words.on_changed(update)
        shauteur_layer.on_changed(update)
        shauteur_head.on_changed(update)
        check.on_clicked(update)

        plt.show()

        return 0


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
