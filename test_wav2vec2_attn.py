import torch
import librosa
from steve import STEVE
from transformers import Wav2Vec2Processor, Wav2Vec2Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load wav2vec 2.0 model
model_name = "facebook/wav2vec2-base-960h"
# processor
processor = Wav2Vec2Processor.from_pretrained(model_name)
# model
model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)

# Create attention hooks
encoder_attn = [None] * len(model.encoder.layers)
for i, block in enumerate(model.encoder.layers):
    block.attention.register_forward_hook(
        lambda _, ins, outs, index=i: encoder_attn.__setitem__(index, outs[1])
    )

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
    speech_, sr_ = librosa.load(path=file, sr=16000)

    mel = processor(speech_, return_tensors="pt",
                        padding="longest", sampling_rate=sr_).input_values.to(DEVICE)

    with torch.no_grad():
        model.eval()
        model(mel, output_hidden_states=True, output_attentions=True)

    spf = len(speech_) / len(encoder_attn[0][0][0])

    attentions.append(encoder_attn.copy())
    spfs.append(spf)

    print("spf:", spf)
    print(len(attentions[0][0]))

# Plot the attentions
steve = STEVE()
steve.plot_attentions(file_name=files, attention=attentions, spf=spfs,
                      alignments=alignment, tg_phone=tg_phn, tg_word=tg_wrd, tg_text=tg_txt,
                      tg_ext="TextGrid", debug=False)
