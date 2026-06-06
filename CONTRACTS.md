# Module Contracts — Kannada Farmer Voice Assistant

Pipeline flow:
```
Mic → [module2_asr] → Kannada text
                           │
                    [module3_advisor] ← optional crop photo → [module1_disease]
                           │
                    Kannada advice text
                           │
                    [module4_tts] → audio bytes → Speaker
```

All public functions are **synchronous** unless noted `async`. Each module
exposes exactly the function(s) described here; internal helpers are private.

---

## module1_disease — Crop Disease Detector

**File:** `module1_disease/detector.py`

```python
def predict_disease(image_path: str) -> dict:
    """
    Classify a crop leaf photo using EfficientNet-B0 trained on PlantVillage.

    Args:
        image_path: Path to a JPEG/PNG crop photo.

    Returns:
        {
            "label":      str,    # PlantVillage class, e.g. "Tomato___Late_blight"
            "confidence": float,  # softmax probability in [0, 1]
        }

    Raises:
        FileNotFoundError: if image_path or model weights are missing.
    """
```

**Training:** `module1_disease/train.py` — EfficientNet-B0, AMP (mixed precision),
batch 32, CosineAnnealingLR, AdamW, label smoothing 0.1.
Outputs: `models/disease_model.pt`, `models/disease_classes.json`.

---

## module2_asr — Automatic Speech Recognition (Kannada)

**File:** `module2_asr/transcriber.py`

```python
def transcribe(audio_path: str) -> str:
    """
    Transcribe a Kannada audio file to Kannada Unicode text.

    Accepts any format torchaudio supports (WAV, MP3, FLAC, OGG…).
    Auto-resamples to 16 kHz mono before inference.
    Always runs on CPU (VRAM reserved for disease classifier).

    Args:
        audio_path: Path to the audio file.

    Returns:
        Kannada Unicode string, e.g. "ನನ್ನ ಟೊಮೆಟೊ ಗಿಡಕ್ಕೆ ರೋಗ ಬಂದಿದೆ"

    Raises:
        FileNotFoundError: if audio_path does not exist.
        ImportError:       if NeMo is not installed.
    """
```

**Model:** `ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large` (NeMo, ~500 MB download on first use).
**Decoder:** CTC by default (fast); switch `_model.cur_decoder = "rnnt"` for higher accuracy.

**NeMo install (one-time):**
```
pip install "nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@nemo-v2"
```

**Test script:** `module2_asr/test_asr.py`
```
python module2_asr/test_asr.py --record --seconds 6
python module2_asr/test_asr.py path/to/clip.wav --decoder rnnt
```

---

## module3_advisor — Crop Advisor (Offline KB)

**File:** `module3_advisor/advisor.py`

```python
def generate_advice(
    question: str | None = None,
    disease_label: str | None = None,
) -> str:
    """
    Generate Kannada farming advice, fully offline (JSON knowledge base).

    Priority:
      1. disease_label exact match in KB  → full structured advice
      2. disease_label unknown but "healthy" → healthy plant response
      3. disease_label unknown → keyword match on label text
      4. question keyword match            → topic-aware answer
      5. no match                          → generic fallback

    Args:
        question:      Free-text Kannada question (from ASR), or None.
        disease_label: PlantVillage class name (from disease classifier), or None.

    Returns:
        Kannada Unicode string suitable for TTS.
    """
```

**Knowledge base:** `module3_advisor/knowledge_base.json`
Seeded with 5 diseases: Tomato___Late_blight, Tomato___Early_blight,
Potato___Early_blight, Corn_(maize)___Common_rust_, Tomato___Bacterial_spot.
Each entry has: `name_kannada`, `cause`, `treatment`, `prevention`, `keywords`.
Add new diseases by appending entries to the `"diseases"` object — no code changes needed.

---

## module4_tts — Text-to-Speech (Kannada)

**File:** `module4_tts/synthesizer.py`

```python
def synthesize(text: str, out_path: str) -> str:
    """
    Synthesize Kannada text to a WAV file using IndicF5 voice-cloning TTS.

    Uses the bundled reference clip in module4_tts/reference/ for voice.
    Runs on CPU. Output: 24 000 Hz mono WAV.

    Args:
        text:     Kannada Unicode string to speak.
        out_path: Destination WAV path (parent dirs created automatically).

    Returns:
        out_path (for chaining).

    Raises:
        ValueError:  if text is empty.
        RuntimeError: if IndicF5 synthesis fails.
    """

def use_custom_reference(audio_path: str, transcript: str) -> None:
    """
    Hot-swap the reference voice without restarting.

    Args:
        audio_path: 24 kHz mono WAV (3–10 s recommended).
        transcript: Exact Kannada text spoken in audio_path.
    """
```

**Model:** `ai4bharat/IndicF5` loaded with `trust_remote_code=True` (~1 GB download on first use).
**Reference audio:** `module4_tts/reference/kannada_ref.wav` (auto-generated 4-second synthetic
placeholder on first run; replace with a real recording for better voice quality — see `module4_tts/reference/README.md`).
**Dependencies installed:** `einops`, `vocos`, `huggingface_hub`, `soundfile`, `transformers`.

---

## app — Pipeline Orchestrator

**File:** `app/pipeline.py`

```python
def run_pipeline(
    audio_path: str | None = None,
    image_path: str | None = None,
    *,
    speak_response: bool = True,
) -> PipelineResult:
    """
    End-to-end pipeline: ASR → (disease detection) → advisor → TTS.

    Args:
        audio_path:     Path to farmer's question audio (or None to record live).
        image_path:     Optional crop photo path (skips module1 if None).
        speak_response: If True, play the TTS audio before returning.

    Returns:
        PipelineResult(
            query_text: str,
            disease:    DiseaseResult | None,
            advice_text: str,
            audio_bytes: bytes,
        )
    """
```

**File:** `app/api.py` — FastAPI app exposing the pipeline over HTTP (optional, for UI or remote use).

---

## Shared exceptions

Defined in `app/exceptions.py` and imported by all modules:

| Exception          | Raised by              |
|--------------------|------------------------|
| `DetectionError`   | module1_disease        |
| `TranscriptionError` | module2_asr          |
| `AudioDeviceError` | module2_asr, module4_tts |
| `AdvisorError`     | module3_advisor        |
| `SynthesisError`   | module4_tts            |
