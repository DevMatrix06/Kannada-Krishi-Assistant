"""
Kannada ASR using AI4Bharat IndicConformer (NeMo-based).

Model: ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large
Runs on CPU intentionally — keeps VRAM free for the disease classifier.

NeMo installation (one-time, from project root):
    pip install "nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@nemo-v2"

If that fails on Windows, clone and install:
    git clone https://github.com/AI4Bharat/NeMo.git
    cd NeMo && git checkout nemo-v2
    pip install -e ".[asr]"
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# NeMo is installed as an editable package from ./NeMo/.
# Add it explicitly so the import works regardless of whether the venv
# .pth-based finder has been activated (e.g. when Gradio spawns workers).
_NEMO_SRC = Path(__file__).parent.parent / "NeMo"
if _NEMO_SRC.exists() and str(_NEMO_SRC) not in sys.path:
    sys.path.insert(0, str(_NEMO_SRC))

# Workaround 1: Prevent numba from loading the CUDA driver at import time.
# Several NeMo parts do `from numba import cuda` at module level, which hangs
# on Windows when the CUDA driver takes time to initialise.
# NUMBA_DISABLE_CUDA=1 makes numba.cuda.is_available() return False instantly
# without touching the driver — PyTorch CUDA is unaffected.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

# Workaround 2: Pre-import torchmetrics before NeMo loads it.
# nemo.collections.asr.models.classification_models imports torchmetrics
# deep inside NeMo's import chain. On Windows, loading torchmetrics' C
# extensions in that context causes a fatal crash (exit 5). Loading it
# here first, at top level, avoids the issue.
import torchmetrics  # noqa: F401  (side-effect import)

logger = logging.getLogger(__name__)

_MODEL_ID  = "ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large"
_TARGET_SR = 16_000
_LANG      = "kn"

_model = None   # lazy-loaded on first call


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_nemo():
    try:
        import nemo.collections.asr as nemo_asr  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "AI4Bharat NeMo is not installed.\n"
            "Run: pip install \"nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@nemo-v2\"\n"
            "Or clone and install from source (see module docstring)."
        ) from e


def _load_model():
    global _model
    if _model is not None:
        return

    _require_nemo()
    import nemo.collections.asr as nemo_asr

    logger.info("Loading IndicConformer Kannada ASR model (first call downloads ~500 MB)…")
    model = nemo_asr.models.ASRModel.from_pretrained(_MODEL_ID)
    model.freeze()
    model = model.to("cpu")   # CPU keeps 4 GB VRAM free for disease classifier
    _model = model
    logger.info("ASR model ready.")


def _to_16k_mono_wav(audio_path: str) -> tuple[str, bool]:
    """
    Ensure audio is 16 kHz mono WAV.
    Returns (path, is_temp) where is_temp=True means the file must be deleted after use.
    """
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)

    needs_resample = sr != _TARGET_SR
    needs_mono     = waveform.shape[0] > 1
    needs_convert  = Path(audio_path).suffix.lower() != ".wav"

    if not needs_resample and not needs_mono and not needs_convert:
        return audio_path, False

    if needs_mono:
        waveform = waveform.mean(dim=0, keepdim=True)

    if needs_resample:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=_TARGET_SR)
        waveform  = resampler(waveform)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    torchaudio.save(tmp.name, waveform, _TARGET_SR)
    return tmp.name, True


# ── public API ────────────────────────────────────────────────────────────────

def transcribe(audio_path: str) -> str:
    """
    Transcribe a Kannada audio file to Kannada Unicode text.

    Accepts any format torchaudio can read (WAV, MP3, FLAC, OGG…).
    Internally resamples to 16 kHz mono before passing to the model.
    Always runs on CPU.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Kannada Unicode string, e.g. "ನನ್ನ ತಳ್ಳೆ ಗಿಡಕ್ಕೆ ರೋಗ ಬಂದಿದೆ"

    Raises:
        FileNotFoundError: if the file does not exist.
        ImportError:       if NeMo is not installed.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _load_model()

    wav_path, is_temp = _to_16k_mono_wav(audio_path)
    try:
        _model.cur_decoder = "ctc"   # CTC is faster; switch to "rnnt" for higher accuracy
        results = _model.transcribe([wav_path], batch_size=1, language_id=_LANG)
        # NeMo may return strings or (text, logprob) tuples depending on version
        raw = results[0]
        text = raw[0] if isinstance(raw, (list, tuple)) else raw
        return text.strip()
    finally:
        if is_temp and os.path.exists(wav_path):
            os.unlink(wav_path)
