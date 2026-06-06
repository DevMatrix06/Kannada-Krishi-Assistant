"""
Kannada TTS using AI4Bharat IndicF5 (F5-TTS based, voice-cloning).

IndicF5 is a zero-shot voice-cloning model: every synthesis call needs
a reference WAV + its transcript to set the speaker's voice.
A Kannada reference clip is bundled in module4_tts/reference/:
  - kannada_ref.wav   (downloaded from HuggingFace on first use, or synthetic)
  - ref_text.txt      (transcript of the reference clip)

Replace kannada_ref.wav with your own clean 24 kHz mono Kannada recording
to get a better-sounding, consistent voice.

Runs on CPU — VRAM is reserved for the disease classifier.
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_MODEL_ID    = "ai4bharat/IndicF5"
_SAMPLE_RATE = 24_000
_REF_DIR     = Path(__file__).parent / "reference"
_REF_AUDIO   = _REF_DIR / "kannada_ref.wav"
_REF_TEXT_F  = _REF_DIR / "ref_text.txt"

# Fallback transcript when the reference was synthetically generated
_SYNTHETIC_REF_TEXT = (
    "ನಮಸ್ಕಾರ. ನಾನು ಕನ್ನಡ ಕೃಷಿ ಸಹಾಯಕ. "
    "ನಿಮ್ಮ ಬೆಳೆ ರೋಗದ ಬಗ್ಗೆ ನಾನು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ."
)

_model = None


# ── reference audio management ────────────────────────────────────────────────

def _write_ref_text(text: str) -> None:
    _REF_TEXT_F.write_text(text, encoding="utf-8")


def _read_ref_text() -> str:
    if _REF_TEXT_F.exists():
        return _REF_TEXT_F.read_text(encoding="utf-8").strip()
    return _SYNTHETIC_REF_TEXT


def _generate_synthetic_ref() -> None:
    """
    Create a 4-second speech-like WAV using harmonic synthesis.
    Good enough to make IndicF5 run; replace with a real voice for production.
    """
    duration = 4.0
    t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), dtype=np.float32)

    f0 = 185.0  # Hz — female speech fundamental
    signal = (
        0.50 * np.sin(2 * np.pi * f0 * t)
        + 0.28 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.14 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.08 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    # Syllable-like amplitude envelope (~3 syllables/sec)
    envelope = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    audio = (signal * envelope * 0.72).astype(np.float32)

    sf.write(str(_REF_AUDIO), audio, _SAMPLE_RATE)
    _write_ref_text(_SYNTHETIC_REF_TEXT)

    logger.warning(
        "Synthetic reference audio written to %s.\n"
        "  → Replace with a real 24 kHz mono Kannada WAV for better voice quality.",
        _REF_AUDIO,
    )


def _fetch_hf_reference() -> bool:
    """
    Try to download a real Kannada reference clip from the IndicF5 HuggingFace
    repo (prompts/ folder). Returns True if successful.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    # Candidate filenames — IndicF5 uses ISO 639-1 prefix + speaker code
    candidates = {
        "prompts/KAN_F_HAPPY_00001.wav": "ನಮಸ್ಕಾರ, ನಾನು ಕನ್ನಡ ಭಾಷೆಯ ಸ್ಪೀಕರ್.",
        "prompts/KAN_M_HAPPY_00001.wav": "ನಮಸ್ಕಾರ, ನಾನು ಕನ್ನಡ ಭಾಷೆಯ ಸ್ಪೀಕರ್.",
        "prompts/KAN_F_NEUTRAL_00001.wav": "ಇದು ಒಂದು ಕನ್ನಡ ಧ್ವನಿ ಮಾದರಿ.",
    }
    for filename, transcript in candidates.items():
        try:
            cached = hf_hub_download(repo_id=_MODEL_ID, filename=filename)
            shutil.copy(cached, _REF_AUDIO)
            _write_ref_text(transcript)
            logger.info("Downloaded Kannada reference: %s", filename)
            return True
        except Exception:
            continue

    return False


def _ensure_reference() -> None:
    _REF_DIR.mkdir(parents=True, exist_ok=True)
    if _REF_AUDIO.exists():
        return
    if not _fetch_hf_reference():
        logger.info("No HuggingFace Kannada prompt found — generating synthetic reference.")
        _generate_synthetic_ref()


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model() -> None:
    global _model
    if _model is not None:
        return

    _ensure_reference()

    logger.info("Loading IndicF5 (first run downloads ~1 GB from HuggingFace)…")
    from transformers import AutoModel

    _model = AutoModel.from_pretrained(
        _MODEL_ID,
        trust_remote_code=True,
        # No device_map — stays on CPU by default; VRAM reserved for disease model
    )
    # Explicitly ensure CPU in case the custom code auto-detects GPU
    try:
        import torch
        _model = _model.to(torch.device("cpu"))
    except Exception:
        pass

    logger.info("IndicF5 ready on CPU.")


# ── output normalisation ──────────────────────────────────────────────────────

def _to_numpy(raw) -> np.ndarray:
    """
    IndicF5 returns whatever the underlying F5-TTS forward pass produces.
    Normalise to a 1-D float32 numpy array regardless of exact type.
    """
    import torch

    if isinstance(raw, dict):
        # e.g. {"audio": tensor, ...} or {"waveform": tensor}
        raw = raw.get("audio") or raw.get("waveform") or next(iter(raw.values()))
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().float().numpy()
    arr = np.asarray(raw, dtype=np.float32)
    # Flatten: (1, T) → (T,) or (C, T) → (T,)
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr


# ── public API ────────────────────────────────────────────────────────────────

def synthesize(text: str, out_path: str) -> str:
    """
    Synthesize Kannada text to a WAV file.

    Uses the bundled reference audio in module4_tts/reference/ for voice.
    Runs on CPU. Output sample rate: 24 000 Hz mono.

    Args:
        text:     Kannada Unicode string to speak.
        out_path: Destination file path (will be created, must end in .wav).

    Returns:
        out_path (as a convenience for chaining).

    Raises:
        RuntimeError: if IndicF5 synthesis fails.
    """
    if not text.strip():
        raise ValueError("text must not be empty")

    _load_model()

    ref_text = _read_ref_text()

    try:
        raw = _model(
            text,
            ref_audio_path=str(_REF_AUDIO),
            ref_text=ref_text,
        )
    except Exception as exc:
        raise RuntimeError(f"IndicF5 synthesis failed: {exc}") from exc

    audio = _to_numpy(raw)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, _SAMPLE_RATE)

    duration = len(audio) / _SAMPLE_RATE
    logger.info("Saved %.1f s of audio → %s", duration, out)
    return str(out)


def use_custom_reference(audio_path: str, transcript: str) -> None:
    """
    Hot-swap the reference voice without restarting.

    Args:
        audio_path: Path to a 24 kHz mono WAV (3–10 s recommended).
        transcript: Exact Kannada text spoken in audio_path.
    """
    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    _REF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, _REF_AUDIO)
    _write_ref_text(transcript)
    logger.info("Reference updated. New voice will be used on next synthesize() call.")
