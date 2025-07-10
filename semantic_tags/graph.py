from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Any
from collections import Counter, defaultdict

import networkx as nx


@dataclass
class Nugget:
    id: int
    text: str
    tags: List[str]
    cluster_id: int
    source: Path


@dataclass
class Tag:
    name: str
    count: int = 0


class TagGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_nuggets(self, nuggets: Iterable[Nugget]):
        for nugget in nuggets:
            self.graph.add_node(
                f"nugget_{nugget.id}",
                type="nugget",
                text=nugget.text,
                cluster=nugget.cluster_id,
                source=str(nugget.source),
            )
            for tag in nugget.tags:
                tag_node = f"tag_{tag}"
                self.graph.add_node(tag_node, type="tag")
                self.graph.add_edge(f"nugget_{nugget.id}", tag_node)
                self.graph.nodes[tag_node]["count"] = self.graph.nodes[tag_node].get("count", 0) + 1

    def co_occurrence_edges(self):
        tags = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "tag"]
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1 :]:
                shared = len(set(self.graph.neighbors(t1)).intersection(self.graph.neighbors(t2)))
                if shared:
                    self.graph.add_edge(t1, t2, weight=shared)

    def to_networkx(self) -> nx.Graph:
        return self.graph

    def summary(self) -> Dict[str, Any]:
        """Return a summary of tags, cluster counts and labels."""
        tags = {
            n[4:]: self.graph.nodes[n].get("count", 0)
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "tag"
        }

        cluster_members = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "nugget":
                cluster_members[data["cluster"]].append(node)

        cluster_labels: Dict[str, str | None] = {}
        for cid, nug_nodes in cluster_members.items():
            counter: Counter[str] = Counter()
            for n in nug_nodes:
                for neigh in self.graph.neighbors(n):
                    if self.graph.nodes[neigh].get("type") == "tag":
                        counter[neigh[4:]] += 1
            cluster_labels[str(cid)] = counter.most_common(1)[0][0] if counter else None

        return {
            "tag_counts": tags,
            "cluster_count": len(cluster_members),
            "clusters": cluster_labels,
        }
