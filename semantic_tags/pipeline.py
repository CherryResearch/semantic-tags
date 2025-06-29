from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .ingestion import load_transcripts
from .chunking import split_into_nuggets
from .vectorization import Embedder
from .tagging import HeuristicTagger
from .clustering import choose_k, cluster_embeddings
from .graph import Nugget, TagGraph
from .weaviate_store import WeaviateStore


class Pipeline:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.embedder = Embedder(model_name=model_name, batch_size=batch_size, device=device)
        self.tagger = HeuristicTagger(labels=tags)

    def run(
        self,
        path: Path,
        *,
        summary_path: Optional[Path] = None,
        store: Optional[WeaviateStore] = None,
    ) -> TagGraph:
        texts = load_transcripts(path)
        nuggets: List[str] = []
        sources: List[Path] = []
        for text, rel_path in texts:
            ns = split_into_nuggets(text)
            nuggets.extend(ns)
            sources.extend([rel_path] * len(ns))
        embeddings = self.embedder.embed(nuggets)
        k = choose_k(embeddings)
        labels, _ = cluster_embeddings(embeddings, k)
        tag_lists = self.tagger.tag(nuggets)
        tg = TagGraph()
        nugget_objs = [
            Nugget(i, t, tags, int(label), sources[i])
            for i, (t, tags, label) in enumerate(zip(nuggets, tag_lists, labels))
        ]
        tg.add_nuggets(nugget_objs)
        tg.co_occurrence_edges()
        if summary_path is not None:
            with open(summary_path, "w", encoding="utf-8") as f:
                import json

                json.dump(tg.summary(), f, indent=2)
        if store is not None:
            store.add_tag_graph(tg)
        return tg
