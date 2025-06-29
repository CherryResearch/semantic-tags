import re
from collections import defaultdict
from typing import Dict, Iterable, List


DEFAULT_LABELS = {
    "recipe": [r"\brecipe\b", r"\bcook\b"],
    "anime": [r"\banime\b", r"\bmanga\b"],
}


class HeuristicTagger:
    def __init__(self, patterns: Dict[str, List[str]] = None, labels: List[str] | None = None):
        if labels is not None:
            patterns = {l: [rf"\b{re.escape(l)}\b"] for l in labels}
        self.patterns = {k: [re.compile(p, re.I) for p in v] for k, v in (patterns or DEFAULT_LABELS).items()}

    def tag(self, texts: Iterable[str]) -> List[List[str]]:
        results: List[List[str]] = []
        for text in texts:
            tags = []
            for label, regexes in self.patterns.items():
                if any(r.search(text) for r in regexes):
                    tags.append(label)
            results.append(tags)
        return results
