import json
import re
from collections import Counter
from typing import List, Optional

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

from .graph import TagGraph


def suggest_missing_tags(
    tg: TagGraph, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"
) -> List[str]:
    """Suggest additional tags from the existing graph.

    If ``api_key`` is provided and the ``openai`` package is available, the tag
    counts will be sent to OpenAI to request suggestions. When the key is not
    given or the API call fails, a simple heuristic returns the most common
    tokens that do not already appear as tags.
    """
    summary = tg.summary()
    tag_counts = summary.get("tag_counts", {})

    if api_key and openai is not None:
        try:
            openai.api_key = api_key
            prompt = (
                "Existing tag counts: "
                + json.dumps(tag_counts)
                + ". Suggest up to five additional tags that might be missing. "
                "Return a JSON array of tag names."
            )
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except Exception:
                return [line.strip("- ") for line in content.splitlines() if line.strip()]
        except Exception:
            pass

    # Fallback heuristic
    existing_tags = set(tag_counts)
    tokens: Counter[str] = Counter()
    for node, data in tg.graph.nodes(data=True):
        if data.get("type") == "nugget":
            tokens.update(re.findall(r"\b\w{3,}\b", data.get("text", "").lower()))
    suggestions = [t for t, _ in tokens.most_common() if t not in existing_tags]
    return suggestions[:5]
