"""
Quick test for the Kannada ASR module.

Modes
-----
1. Transcribe an existing file:
       python module2_asr/test_asr.py path/to/audio.wav

2. Record from the microphone, then transcribe:
       python module2_asr/test_asr.py --record
       python module2_asr/test_asr.py --record --seconds 8

Run from the project root with the venv active.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path


def record_clip(seconds: int = 5, sample_rate: int = 16_000) -> str:
    """Record from the default mic. Returns path to a temporary WAV file."""
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit(
            "[ERROR] sounddevice / soundfile not installed.\n"
            "Run: pip install sounddevice soundfile"
        )

    print(f"\nRecording {seconds}s of audio — speak now…", flush=True)
    for i in range(3, 0, -1):
        print(f"  {i}…", flush=True)
        time.sleep(1)
    print("  GO!", flush=True)

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  Done recording.\n")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description="Test Kannada ASR")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("audio_file", nargs="?", help="Path to an existing audio file")
    group.add_argument("--record", action="store_true", help="Record from microphone")
    parser.add_argument("--seconds", type=int, default=5,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--decoder", choices=["ctc", "rnnt"], default="ctc",
                        help="Decoder to use: ctc (faster) or rnnt (more accurate)")
    args = parser.parse_args()

    # Resolve audio path
    if args.record:
        audio_path = record_clip(seconds=args.seconds)
        is_temp = True
    else:
        audio_path = args.audio_file
        is_temp = False
        if not Path(audio_path).exists():
            sys.exit(f"[ERROR] File not found: {audio_path}")

    # Override decoder if requested
    import module2_asr.transcriber as asr
    # Patch decoder choice before loading model
    _orig_transcribe = asr.transcribe

    def _transcribe_with_decoder(path):
        asr._load_model()
        import os, tempfile
        wav_path, tmp = asr._to_16k_mono_wav(path)
        try:
            asr._model.cur_decoder = args.decoder
            results = asr._model.transcribe([wav_path], batch_size=1, language_id="kn")
            raw = results[0]
            return (raw[0] if isinstance(raw, (list, tuple)) else raw).strip()
        finally:
            if tmp and os.path.exists(wav_path):
                os.unlink(wav_path)

    try:
        print(f"Audio   : {audio_path}")
        print(f"Decoder : {args.decoder}")
        print("Loading model (first run downloads ~500 MB)…\n")

        t0 = time.time()
        text = _transcribe_with_decoder(audio_path)
        elapsed = time.time() - t0

        print("=" * 50)
        print(f"  Kannada: {text}")
        print("=" * 50)
        print(f"  (inference took {elapsed:.1f}s on CPU)")

    except ImportError as e:
        sys.exit(f"[ERROR] {e}")
    finally:
        if is_temp:
            import os
            os.unlink(audio_path)


if __name__ == "__main__":
    main()
