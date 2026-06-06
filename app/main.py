"""
Kannada Farmer Voice Assistant — Gradio demo
============================================
Pipeline: mic → ASR → (disease classifier) → advisor → TTS → UI

Each heavy model is loaded inside the pipeline function so only one sits
in memory at a time — important on a 4 GB RTX 3050.

Run from the project root:
    venv\\Scripts\\activate
    python app/main.py
"""

import gc
import logging
import sys
from pathlib import Path
from typing import Generator

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from module1_disease.detector import predict_disease
from module2_asr.transcriber import transcribe
from module3_advisor.advisor import generate_advice
from module4_tts.synthesizer import synthesize

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = Path("output")
_OUT_DIR.mkdir(exist_ok=True)

# Type alias for the 5-tuple every yield must produce
# (status, transcript, disease_label, advice_text, audio_path)
_YieldRow = tuple[str, str, str, str, str | None]


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _empty(status: str) -> _YieldRow:
    """Return a row with only the status field filled."""
    return status, "", "", "", None


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(audio_path: str | None, image_path: str | None) -> Generator[_YieldRow, None, None]:
    """
    Runs the four-module pipeline and yields UI state after each step.
    Every yield is a complete 5-tuple so Gradio never receives a partial update.
    """
    if audio_path is None:
        yield _empty("⚠️ ದಯವಿಟ್ಟು ಮೊದಲು ಮೈಕ್‌ನಲ್ಲಿ ಮಾತನಾಡಿ.")
        return

    # ── Step 1: ASR (NeMo IndicConformer — CPU) ───────────────────────────────
    yield _empty("🎙️  ನಿಮ್ಮ ಮಾತನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುತ್ತಿದ್ದೇನೆ…")
    try:
        question = transcribe(audio_path)
        logger.info("ASR → %r", question)
    except Exception as exc:
        yield _empty(f"❌ ASR ವಿಫಲ: {exc}")
        return
    finally:
        _free_gpu()

    # ── Step 2: Disease detection (EfficientNet-B0 — GPU) ─────────────────────
    disease_label: str | None = None
    disease_badge = "—  (ಫೋಟೊ ಒದಗಿಸಿಲ್ಲ)"

    if image_path is not None:
        yield "🌿  ಎಲೆ ಫೋಟೊ ವಿಶ್ಲೇಷಿಸುತ್ತಿದ್ದೇನೆ…", question, "", "", None
        try:
            result        = predict_disease(image_path)
            disease_label = result["label"]
            conf          = result["confidence"]
            disease_badge = f"{disease_label}  ({conf:.0%})"
            logger.info("Disease → %s  (%.0f%%)", disease_label, conf * 100)
        except Exception as exc:
            logger.warning("Disease detection failed (non-fatal): %s", exc)
            disease_badge = f"⚠️ ರೋಗ ಗುರುತಿಸಲು ವಿಫಲ: {exc}"
        finally:
            _free_gpu()

    # ── Step 3: Advice (offline JSON KB — instant) ────────────────────────────
    yield "💡  ಸಲಹೆ ತಯಾರಿಸುತ್ತಿದ್ದೇನೆ…", question, disease_badge, "", None
    advice = generate_advice(question=question, disease_label=disease_label)
    logger.info("Advice → %d chars", len(advice))

    # ── Step 4: TTS (IndicF5 — CPU) ───────────────────────────────────────────
    yield "🔊  ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ…", question, disease_badge, advice, None
    audio_out: str | None = None
    try:
        audio_out = str(_OUT_DIR / "advice.wav")
        synthesize(advice, audio_out)
    except Exception as exc:
        logger.error("TTS failed: %s", exc)
        audio_out = None
    finally:
        _free_gpu()

    yield "✅  ಮುಗಿದಿದೆ", question, disease_badge, advice, audio_out


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = """
.status-ok  { color: #2d6a2d; font-weight: bold; }
.status-err { color: #b00020; font-weight: bold; }
"""

_HEADER = """
<div style="text-align:center; padding: 8px 0 4px;">
  <h1 style="font-size:2rem; margin:0;">🌾 ಕನ್ನಡ ಕೃಷಿ ಸಹಾಯಕ</h1>
  <p style="color:#555; margin:4px 0 0;">Kannada Farmer Voice Assistant</p>
</div>
"""

_HOW_TO = """
**ಹೇಗೆ ಬಳಸಬೇಕು:**
1. 🎙️ ಮೈಕ್ ಬಟನ್ ಒತ್ತಿ ಕನ್ನಡದಲ್ಲಿ ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಕೇಳಿ
2. 📷 ಬೆಳೆ ಎಲೆಯ ಫೋಟೊ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ *(ಐಚ್ಛಿಕ)*
3. 🚜 **ಸಲಹೆ ಪಡೆಯಿರಿ** ಒತ್ತಿ
"""

with gr.Blocks(
    title="ಕನ್ನಡ ಕೃಷಿ ಸಹಾಯಕ",
) as demo:

    gr.HTML(_HEADER)
    gr.Markdown(_HOW_TO)
    gr.Markdown("---")

    with gr.Row(equal_height=False):

        # ── Left: inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📥 ನಿಮ್ಮ ಪ್ರಶ್ನೆ")

            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಿ  (tap mic or upload WAV)",
            )
            image_in = gr.Image(
                type="filepath",
                label="ಬೆಳೆ ಎಲೆಯ ಫೋಟೊ  (ಐಚ್ಛಿಕ / optional)",
                height=200,
            )
            submit_btn = gr.Button(
                "🚜  ಸಲಹೆ ಪಡೆಯಿರಿ",
                variant="primary",
                size="lg",
            )

        # ── Right: outputs ────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📤 ಫಲಿತಾಂಶ")

            status_out = gr.Textbox(
                label="ಸ್ಥಿತಿ",
                interactive=False,
                max_lines=1,
            )
            transcript_out = gr.Textbox(
                label="ನೀವು ಕೇಳಿದ್ದು  (ASR transcript)",
                interactive=False,
                lines=2,
            )
            disease_out = gr.Textbox(
                label="ರೋಗ ಗುರುತು  (Disease label + confidence)",
                interactive=False,
                max_lines=2,
            )
            advice_out = gr.Textbox(
                label="ಸಲಹೆ  (Kannada advice)",
                interactive=False,
                lines=6,
            )
            audio_out = gr.Audio(
                label="🔊 ಧ್ವನಿ ಸಲಹೆ",
                type="filepath",
                interactive=False,
                autoplay=True,
            )

    # ── Event ─────────────────────────────────────────────────────────────────
    submit_btn.click(
        fn=run_pipeline,
        inputs=[audio_in, image_in],
        outputs=[status_out, transcript_out, disease_out, advice_out, audio_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="green"),
        css=_CSS,
    )
