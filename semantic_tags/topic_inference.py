from __future__ import annotations

from typing import Dict, List

import re
from collections import Counter


def infer_cluster_tags(
    nuggets: List[str], labels: List[int], top_n: int = 2, api_key: str | None = None
) -> Dict[int, str]:
    """Return a short label for each cluster.

    If ``api_key`` is provided, the OpenAI API will be queried. Otherwise a simple
    word-frequency heuristic is used.
    """
    n_clusters = max(labels) + 1 if labels else 0
    result: Dict[int, str] = {}
    for cid in range(n_clusters):
        texts = [t for t, l in zip(nuggets, labels) if l == cid]
        if not texts:
            continue
        if api_key:
            try:
                import openai

                openai.api_key = api_key
                prompt = (
                    "Provide a 1-2 word topic label for the following text:\n" + " ".join(texts)
                )
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
                label = resp["choices"][0]["message"]["content"].strip()
                result[cid] = label
                continue
            except Exception:
                pass
        tokens = re.findall(r"\b\w{3,}\b", " ".join(texts).lower())
        counts = Counter(tokens)
        if counts:
            label = " ".join(t for t, _ in counts.most_common(top_n))
        else:
            label = f"cluster_{cid}"
        result[cid] = label
    return result
