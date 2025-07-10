from __future__ import annotations

import re
from typing import List, Tuple

SPEAKER_RE = re.compile(r"^(\w+):\s+(.*)")

POSITIVE_WORDS = {"love", "like", "good", "great", "awesome", "happy"}
NEGATIVE_WORDS = {"hate", "bad", "terrible", "sad", "awful"}


def diarize_and_chunk(text: str) -> List[Tuple[str, str]]:
    """Return a list of (chunk_text, speaker) tuples."""
    chunks: List[Tuple[str, str]] = []
    current_speaker = "Unknown"
    buffer: List[str] = []
    for line in text.splitlines():
        m = SPEAKER_RE.match(line)
        if m:
            if buffer:
                chunks.append((" ".join(buffer).strip(), current_speaker))
                buffer = []
            current_speaker = m.group(1)
            buffer.append(m.group(2))
        else:
            buffer.append(line)
    if buffer:
        chunks.append((" ".join(buffer).strip(), current_speaker))
    return chunks


def detect_emotion(text: str) -> str:
    """Roughly classify emotion from text."""
    tokens = {t.strip(".,!?;:").lower() for t in text.split()}
    if tokens & POSITIVE_WORDS:
        return "positive"
    if tokens & NEGATIVE_WORDS:
        return "negative"
    return "neutral"
