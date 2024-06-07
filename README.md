# STEVE (Speech Transformer Exploratory Visual Environment)

![STEVE screenshot annotated.](/assets/images/STEVE_explained.png)

## Developers/Authors
Erfan A. Shams, Julie Carson-Berndsen

## Publication
TBA

## Installation and Usage
Clone the repository using the command below:
```
git clone https://github.com/erfanashams/steve.git
```

Navigate to w2v2viz folder and install the requirements:

```
cd steve
```

```
pip install -r requirements.txt
```

### How to use

1. Extract the self-attention weights in the following format: N&times;1&times;M&times;K&times;Q, where N is the number of layers, M is the number of self-attention heads, K and Q are the Key and Query dimensions.
For example the code below allows self-attention head extraction from whipsr-base model and stores them in `encoder_attn` variable after inference:

```angular2html
import torch
import librosa
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
```
Inference steps for whisper are given below:

```angular2html
file = "path_to_your_wav_file"
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
```
The audio sample per frame `spf` is also needed to be calculated or extracted from the model parameters.
Refer to the paper for this step. For all whisper models spf is 320.

```angular2html
spf = 320
```
2. Import and run the visualiser using the required ad optional parameters:

```angular2html
from steve import STEVE


steve = STEVE()
steve.plot_attentions(file_name=[file], attention=[encoder_attn], spf=[spf],
                      alignments="", tg_phone=None, tg_word=None, tg_text=None,
                      tg_ext="textgrid", debug=False)
```
Required parameters:
+ file_name: string literal of the wav file or a list of files.
+ attention: the extracted attentions or a list of extracted attentions mentioned above.
+ spf: audio sample per frame of the model.

Optional parameters
- alignments: "timit" or "textgrid" if alignments in any of the given formats are available.
- tg_phone: textgrid phone tier name or TIMIT style phonetic alignment file extension (e.g., phn).
- tg_word: textgrid word tier name or TIMIT style word alignment file extension (e.g., wrd).
- tg_phone: textgrid text tier name or TIMIT style text file extension (e.g., txt).
- tg_ext: extension of the textgrid files (e.g., TextGrid).
- debug: display various steps during the visualisation process.

> [!TIP]
> Complete examples for Whisper (`text_whisper_attn.py`) and wav2vec 2.0 (`test_wav2vec2_attn.py`) with annotations are available in the mentioned files.

### Cite as

```angular2html
TBA
```