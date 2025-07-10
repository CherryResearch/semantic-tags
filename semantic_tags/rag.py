import json
from typing import List

import openai

from .graph import TagGraph


def suggest_missing_tags(tg: TagGraph, api_key: str, model: str = "gpt-3.5-turbo") -> List[str]:
    """Use an LLM to suggest additional tags.

    The function sends the current tag counts to OpenAI and asks for up to five
    missing tags. The response should be a JSON array of tag names. If parsing
    fails, the text will be split by newlines.
    """
    openai.api_key = api_key
    summary = tg.summary()
    tag_counts = summary.get("tag_counts", {})
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
