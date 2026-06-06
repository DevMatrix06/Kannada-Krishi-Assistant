"""
Fully-offline Kannada crop advisor backed by a local JSON knowledge base.

No network calls. No LLM. Instant responses.
"""

import json
from pathlib import Path

_KB_PATH = Path(__file__).parent / "knowledge_base.json"
_kb: dict | None = None


def _load() -> None:
    global _kb
    if _kb is None:
        with open(_KB_PATH, encoding="utf-8") as f:
            _kb = json.load(f)


def _format_full(entry: dict) -> str:
    """Return a flowing Kannada paragraph suitable for TTS."""
    return (
        f"{entry['name_kannada']} ರೋಗ ಕಂಡುಬಂದಿದೆ. "
        f"{entry['cause']}. "
        f"ಚಿಕಿತ್ಸೆಗಾಗಿ, {entry['treatment']}. "
        f"ಭವಿಷ್ಯದಲ್ಲಿ ತಡೆಗಟ್ಟಲು, {entry['prevention']}."
    )


def _format_topic(entry: dict, topic: str) -> str:
    """Return only the section the farmer asked about."""
    if topic == "cause":
        return f"{entry['name_kannada']} ರೋಗದ ಕಾರಣ: {entry['cause']}."
    if topic == "treatment":
        return f"{entry['name_kannada']} ರೋಗಕ್ಕೆ ಚಿಕಿತ್ಸೆ: {entry['treatment']}."
    if topic == "prevention":
        return f"{entry['name_kannada']} ರೋಗ ತಡೆಗಟ್ಟಲು: {entry['prevention']}."
    return _format_full(entry)


def _detect_topic(question: str) -> str | None:
    """Return 'cause', 'treatment', 'prevention', or None."""
    for topic, keywords in _kb["topic_keywords"].items():
        if any(kw in question for kw in keywords):
            return topic
    return None


def _score_diseases(text: str) -> str | None:
    """
    Return the disease label with the most keyword hits in `text`, or None.
    Ties broken by label order (deterministic).
    """
    best_label, best_score = None, 0
    for label, entry in _kb["diseases"].items():
        score = sum(1 for kw in entry["keywords"] if kw in text)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label if best_score > 0 else None


def generate_advice(
    question: str | None = None,
    disease_label: str | None = None,
) -> str:
    """
    Generate Kannada farming advice, fully offline.

    Priority order:
      1. disease_label found in KB              → full structured advice
      2. disease_label given but unknown        → keyword match on the label text
      3. question given                         → keyword match; topic-aware answer
      4. nothing matched                        → generic fallback

    Args:
        question:      Free-text Kannada question from ASR, or None.
        disease_label: PlantVillage class name from the disease classifier, or None.

    Returns:
        Kannada Unicode string ready for TTS.
    """
    _load()

    topic = _detect_topic(question) if question else None

    # ── 1. Direct disease label lookup ────────────────────────────────────────
    if disease_label:
        # Exact match
        if disease_label in _kb["diseases"]:
            entry = _kb["diseases"][disease_label]
            return _format_topic(entry, topic) if topic else _format_full(entry)

        # Healthy label — no disease to advise on
        if "healthy" in disease_label.lower():
            return _kb["general_advice"]["healthy"]

        # Unknown label — try to match by keywords within the label string
        matched = _score_diseases(disease_label.lower())
        if matched:
            entry = _kb["diseases"][matched]
            return _format_topic(entry, topic) if topic else _format_full(entry)

        # Label not in KB at all
        return _kb["general_advice"]["unknown_disease"]

    # ── 2. Free-text question keyword matching ────────────────────────────────
    if question:
        matched = _score_diseases(question)
        if matched:
            entry = _kb["diseases"][matched]
            return _format_topic(entry, topic) if topic else _format_full(entry)

    # ── 3. Nothing matched ────────────────────────────────────────────────────
    return _kb["general_advice"]["no_match"]
