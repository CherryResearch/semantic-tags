from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .ingestion import load_transcripts
from .chunking import split_into_nuggets
from .diarization import diarize_and_chunk, detect_emotion
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
        tag_file: Optional[Path] = None,
    ):
        self.embedder = Embedder(model_name=model_name, batch_size=batch_size, device=device)

        if tags is None and tag_file is not None and Path(tag_file).exists():
            with open(tag_file, "r", encoding="utf-8") as f:
                tags = [line.strip() for line in f if line.strip()]

        self.tagger = HeuristicTagger(labels=tags)

    def run(
        self,
        path: Path,
        *,
        summary_path: Optional[Path] = None,
        store: Optional[WeaviateStore] = None,
        infer_topics: bool = False,
        topic_api_key: Optional[str] = None,
    ) -> TagGraph:
        texts = load_transcripts(path)
        nuggets: List[str] = []
        sources: List[Path] = []
        speakers: List[str] = []
        emotions: List[str] = []
        for text, rel_path in texts:
            for chunk, speaker in diarize_and_chunk(text):
                ns = split_into_nuggets(chunk)
                for n in ns:
                    nuggets.append(n)
                    sources.append(rel_path)
                    speakers.append(speaker)
                    emotions.append(detect_emotion(n))
        embeddings = self.embedder.embed(nuggets)
        k = choose_k(embeddings)
        labels, _ = cluster_embeddings(embeddings, k)
        tag_lists = self.tagger.tag(nuggets)

        if infer_topics:
            from .topic_inference import infer_cluster_tags

            cluster_tags = infer_cluster_tags(nuggets, labels, api_key=topic_api_key)
            tag_lists = [
                tags + [cluster_tags.get(int(label), f"cluster_{label}")]
                for tags, label in zip(tag_lists, labels)
            ]
        tg = TagGraph()
        nugget_objs = [
            Nugget(i, t, tags, int(label), sources[i], spk, emo)
            for i, (t, tags, label, spk, emo) in enumerate(
                zip(nuggets, tag_lists, labels, speakers, emotions)
            )
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
