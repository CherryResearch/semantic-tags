import re
from typing import Iterable, List


SENTENCE_RE = re.compile(r"(?<=[.!?]) +")


def split_into_nuggets(text: str, max_tokens: int = 128) -> List[str]:
    """Split raw text into semantically coherent nuggets."""
    sentences = SENTENCE_RE.split(text)
    nuggets = []
    current = []
    tokens = 0
    for sent in sentences:
        sent_tokens = len(sent.split())
        if tokens + sent_tokens > max_tokens and current:
            nuggets.append(" ".join(current))
            current = []
            tokens = 0
        current.append(sent)
        tokens += sent_tokens
    if current:
        nuggets.append(" ".join(current))
    return nuggets
