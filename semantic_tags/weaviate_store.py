import json
from typing import Dict

import weaviate

from .graph import TagGraph


class WeaviateStore:
    """Simple wrapper to persist TagGraph objects in Weaviate."""

    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.Client(url)
        self.init_schema()

    def init_schema(self) -> None:
        schema = self.client.schema.get()
        classes = {c["class"] for c in schema.get("classes", [])}
        if "Nugget" not in classes:
            self.client.schema.create_class(
                {
                    "class": "Nugget",
                    "properties": [
                        {"name": "text", "dataType": ["text"]},
                        {"name": "cluster", "dataType": ["int"]},
                        {"name": "tags", "dataType": ["text[]"]},
                    ],
                }
            )
        if "Tag" not in classes:
            self.client.schema.create_class(
                {
                    "class": "Tag",
                    "properties": [
                        {"name": "name", "dataType": ["text"]},
                        {"name": "count", "dataType": ["int"]},
                    ],
                }
            )

    def add_tag_graph(self, tg: TagGraph) -> None:
        for node, data in tg.graph.nodes(data=True):
            if data.get("type") == "nugget":
                tags = [n[4:] for n in tg.graph.neighbors(node) if n.startswith("tag_")]
                self.client.data_object.create(
                    {"text": data["text"], "cluster": data["cluster"], "tags": tags},
                    "Nugget",
                )
            elif data.get("type") == "tag":
                self.client.data_object.create(
                    {"name": node[4:], "count": data.get("count", 0)}, "Tag"
                )

    def save_summary(self, tg: TagGraph, path: str) -> None:
        summary = tg.summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
