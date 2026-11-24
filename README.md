# Kannada Farmer Voice Assistant

A voice-first crop advisory assistant for Kannada-speaking farmers. Ask a
question out loud, optionally attach a photo of an affected leaf, and get
spoken advice back in Kannada.

## Pipeline

```
Mic → [module2_asr] → Kannada text
                           │
                    [module3_advisor] ← optional crop photo → [module1_disease]
                           │
                    Kannada advice text
                           │
                    [module4_tts] → audio → Speaker
```

The whole flow is wired together behind a Gradio UI in `app/main.py`.

## Modules

| Module | File | What it does |
|---|---|---|
| `module1_disease` | `detector.py` | Classifies a crop leaf photo with EfficientNet-B0 trained on PlantVillage |
| `module2_asr` | `transcriber.py` | Transcribes Kannada speech using Whisper |
| `module3_advisor` | `advisor.py` | Generates advice from the transcript, the disease label and `knowledge_base.json` |
| `module4_tts` | `synthesizer.py` | Synthesises Kannada speech from the advice text |

Each module's public interface is specified in [CONTRACTS.md](CONTRACTS.md).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # for Whisper API fallback
```

## Running

```bash
python app/main.py
```

This launches the Gradio interface. `predict.py` runs the disease classifier
on its own if you only need that step.

## Note on memory

Each model is loaded inside the pipeline function and released afterwards, so
only one sits in GPU memory at a time. This keeps the assistant usable on a
4 GB card (developed against an RTX 3050).

## Licence

MIT — see [LICENSE](LICENSE).
