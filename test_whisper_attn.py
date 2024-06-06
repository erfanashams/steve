import torch
import librosa
from steve import STEVE
import whisper
from whisper.tokenizer import get_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_size = "base"
model = whisper.load_model(model_size).to(DEVICE)

# install hooks on the encoder attention layers to retrieve the attention weights
encoder_attn = [None] * model.dims.n_audio_layer
for i, block in enumerate(model.encoder.blocks):
    block.attn.register_forward_hook(
        lambda _, ins, outs, index=i: encoder_attn.__setitem__(index, outs[-1])
    )

# Load Whisper tokenizer
tokenizer = get_tokenizer(model.is_multilingual, language='en')

# Define file/files list
# files = [f"TextGrid_sample/test_mono_channel.wav"]
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
            tokenizer.timestamp_begin,
        ]
    ).to(DEVICE)

    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(speech_)).to(DEVICE)
    with torch.no_grad():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))

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
