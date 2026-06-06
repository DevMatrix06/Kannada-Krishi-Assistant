# TTS Reference Voice

IndicF5 is a voice-cloning model. Every synthesis call uses the files in
this folder to set the speaker's voice characteristics.

## Files

| File | Description |
|---|---|
| `kannada_ref.wav` | Reference audio clip (24 kHz mono WAV, 3–10 s) |
| `ref_text.txt` | Exact Kannada transcript of `kannada_ref.wav` |

On first run `synthesizer.py` will:
1. Try to download a real Kannada sample from the IndicF5 HuggingFace repo.
2. If offline or no Kannada sample exists there, generate a synthetic placeholder.

## Replacing with your own voice

Record yourself reading any clear Kannada sentence (5–8 seconds is ideal).
Save it as 24 kHz mono WAV, then either:

**Option A — copy manually:**
```
ffmpeg -i your_recording.mp3 -ar 24000 -ac 1 module4_tts/reference/kannada_ref.wav
# edit module4_tts/reference/ref_text.txt to match what you said
```

**Option B — use the helper (Python):**
```python
from module4_tts.synthesizer import use_custom_reference
use_custom_reference("your_recording.wav", "ನಿಮ್ಮ ಮಾತಿನ ಪಾಠ ಇಲ್ಲಿ ಬರೆಯಿರಿ.")
```

The better the reference recording (quiet room, clear speech, no echo),
the more natural the TTS output will sound.
