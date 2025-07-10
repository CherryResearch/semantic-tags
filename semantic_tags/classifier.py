from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .graph import TagGraph


def train_tag_classifier(graph: TagGraph) -> Optional[Tuple[TfidfVectorizer, LogisticRegression]]:
    """Train a simple classifier from tagged nuggets.

    Returns ``None`` if there are no labelled nuggets.
    """
    texts: List[str] = []
    labels: List[str] = []
    for node, data in graph.graph.nodes(data=True):
        if data.get("type") == "nugget":
            tag_neighbors = [
                n for n in graph.graph.neighbors(node)
                if graph.graph.nodes[n].get("type") == "tag"
            ]
            if tag_neighbors:
                labels.append(tag_neighbors[0][4:])
                texts.append(data["text"])
    if not texts:
        return None

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=100)
    clf.fit(X, labels)
    return vec, clf
