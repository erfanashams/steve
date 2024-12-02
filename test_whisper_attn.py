"""
This test file is for the newer versions of Whisper (v20240930 and newer).
It should work on the older versions as well.
"""

import torch
import librosa
from steve import STEVE
import whisper
from whisper.tokenizer import get_tokenizer

print(whisper.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_size = "base"
model = whisper.load_model(model_size).to(DEVICE)

if model_size == "large-v3":
    n_mels = 128
else:
    n_mels = 80

# create an empty list for the encoder attentions and install hooks on the QKs to retrieve the attention weights
encoder_attn = [None] * model.dims.n_audio_layer

encoder_attn_q = [None] * model.dims.n_audio_layer
for i, block in enumerate(model.encoder.blocks):
    block.attn.query.register_forward_hook(
        lambda _, ins, outs, index=i: encoder_attn_q.__setitem__(index, outs)
    )

encoder_attn_k = [None] * model.dims.n_audio_layer
for i, block in enumerate(model.encoder.blocks):
    block.attn.key.register_forward_hook(
        lambda _, ins, outs, index=i: encoder_attn_k.__setitem__(index, outs)
    )

# Load Whisper tokenizer
tokenizer = get_tokenizer(model.is_multilingual, language='en')

# Define file/files list
# files = [f"TextGrid_sample/arctic_a0001.wav"]
# tg_phn, tg_wrd, tg_txt = ["phones", "words", None]
# alignment = "textgrid"

files = ["TIMIT_sample/LDC93S1.wav"]
tg_phn, tg_wrd, tg_txt = ["phn", "wrd", "txt"]
alignment = "timit"

# Extract attentions
attentions = []
spfs = []
for file in files:
    print(file)
    speech_, sr_ = librosa.load(path=file, sr=16000)
    speech_ = torch.from_numpy(speech_).float()

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
        ]
    ).to(DEVICE)

    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(speech_), n_mels=n_mels).to(DEVICE)
    with torch.no_grad():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))

    # calculate the self-attention weights from QKs
    for i_l in range(model.dims.n_audio_layer):
        i_h = 0
        print(i_l)
        attention_weights = torch.zeros(1, model.dims.n_audio_head, encoder_attn_q[0].shape[1],
                                        encoder_attn_q[0].shape[1]).to(DEVICE)
        for i_h in range(model.dims.n_audio_head):
            print(" ", i_h * 64, i_h * 64 + 64)
            Q = encoder_attn_q[i_l][:, :, i_h * 64:i_h * 64 + 64]
            K = encoder_attn_k[i_l][:, :, i_h * 64:i_h * 64 + 64]

            attention_weights[0, i_h] = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        encoder_attn[i_l] = attention_weights

    spf = whisper.audio.HOP_LENGTH * 2

    attentions.append(encoder_attn.copy())
    spfs.append(spf)

    print("spf:", spf)
    # print(len(attentions[0][0]))

# Plot the attentions
steve = STEVE()
steve.plot_attentions(file_name=files, attention=attentions, spf=spfs,
                      alignments=alignment, tg_phone=tg_phn, tg_word=tg_wrd, tg_text=tg_txt,
                      tg_ext="textgrid", debug=False)
