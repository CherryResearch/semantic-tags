import sys
import types

# Provide dummy numpy and sentence_transformers modules to avoid heavy dependencies
sys.modules["numpy"] = types.ModuleType("numpy")
sys.modules["numpy"].ndarray = list
sys.modules["numpy"].random = types.SimpleNamespace(
    rand=lambda *a, **k: [[0] * 2 for _ in range(a[0])]
)


st_module = types.ModuleType("sentence_transformers")
st_module.SentenceTransformer = lambda *a, **k: types.SimpleNamespace(
    encode=lambda texts, batch_size=32, show_progress_bar=False: [[0] * 2 for _ in texts],
    device=k.get("device", "cpu"),
    model_name=a[0] if a else "model",

)
sys.modules["sentence_transformers"] = st_module

# Stub sklearn modules used by clustering
sklearn_mod = types.ModuleType("sklearn")
cluster_mod = types.ModuleType("sklearn.cluster")
cluster_mod.KMeans = lambda *a, **k: types.SimpleNamespace(fit_predict=lambda X: [0] * len(X))
metrics_mod = types.ModuleType("sklearn.metrics")
metrics_mod.silhouette_score = lambda X, labels: 0.0
sklearn_mod.cluster = cluster_mod
sklearn_mod.metrics = metrics_mod
sys.modules["sklearn"] = sklearn_mod
sys.modules["sklearn.cluster"] = cluster_mod
sys.modules["sklearn.metrics"] = metrics_mod

# Minimal networkx stub used by TagGraph
nx_mod = types.ModuleType("networkx")


class Nodes(dict):
    def __call__(self, data=False):
        return list(self.items()) if data else list(self.keys())


class Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self.nodes = Nodes(self._nodes)

    def add_node(self, node, **attrs):
        self.nodes.setdefault(node, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        self._edges.setdefault((u, v), {}).update(attrs)

    def neighbors(self, node):
        neigh = []
        for u, v in self._edges:
            if u == node:
                neigh.append(v)
            elif v == node:
                neigh.append(u)
        return neigh

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return len(self._edges)


nx_mod.Graph = Graph
sys.modules["networkx"] = nx_mod

# Stub weaviate client used by Pipeline
weaviate_mod = types.ModuleType("weaviate")
weaviate_mod.Client = lambda *a, **k: types.SimpleNamespace(
    schema=types.SimpleNamespace(get=lambda: {"classes": []}),
    data_object=types.SimpleNamespace(create=lambda *args, **kwargs: None),
)
sys.modules["weaviate"] = weaviate_mod

# Stub openai module used by rag functionality
openai_mod = types.ModuleType("openai")
openai_mod.ChatCompletion = types.SimpleNamespace(
    create=lambda **k: {"choices": [{"message": {"content": "[]"}}]}
)
sys.modules["openai"] = openai_mod

# Stub sklearn modules used by classifier
linear_mod = types.ModuleType("sklearn.linear_model")
linear_mod.LogisticRegression = lambda *a, **k: types.SimpleNamespace(
    fit=lambda X, y: None,
    classes_=["recipe"],
)
fe_text_mod = types.ModuleType("sklearn.feature_extraction.text")
fe_text_mod.TfidfVectorizer = lambda *a, **k: types.SimpleNamespace(
    fit_transform=lambda texts: texts,
    transform=lambda texts: texts,
)
fe_mod = types.ModuleType("sklearn.feature_extraction")
fe_mod.text = fe_text_mod
sys.modules["sklearn.linear_model"] = linear_mod
sys.modules["sklearn.feature_extraction"] = fe_mod
sys.modules["sklearn.feature_extraction.text"] = fe_text_mod

from semantic_tags import pipeline as pipeline_mod
from semantic_tags.pipeline import Pipeline


class DummyEmbedder:
    def embed(self, texts):
        # return simple deterministic vectors without numpy
        return [[float(i)] * 2 for i in range(len(texts))]


def test_pipeline_run(tmp_path):
    (tmp_path / "a.md").write_text("This recipe is great. I love to cook.")
    (tmp_path / "b.md").write_text("Anime is a popular genre of manga.")

    pipeline = Pipeline()
    pipeline.embedder = DummyEmbedder()
    # Patch clustering functions to avoid heavy dependencies
    pipeline_mod.choose_k = lambda embeddings, k_min=2, k_max=None: 2
    pipeline_mod.cluster_embeddings = lambda embeddings, k: (list(range(len(embeddings))), None)
    graph = pipeline.run(tmp_path)

    assert graph.graph.number_of_nodes() > 0
    assert graph.graph.number_of_edges() > 0
    nug_node = next(n for n, d in graph.graph.nodes(data=True) if d.get("type") == "nugget")
    nug_data = graph.graph.nodes[nug_node]
    assert nug_data.get("speaker") is not None
    assert nug_data.get("emotion") in {"positive", "negative", "neutral"}
    summary = graph.summary()
    assert summary["clusters"] == {"0": "recipe", "1": "anime"}
    assert summary["cluster_count"] == 2


def test_suggest_missing_tags_openai():
    from semantic_tags.graph import TagGraph
    from semantic_tags.rag import suggest_missing_tags

    tg = TagGraph()
    tg.graph.add_node("tag_recipe", type="tag", count=2)
    tg.graph.add_node("tag_anime", type="tag", count=1)

    suggestions = suggest_missing_tags(tg, "dummy")
    assert suggestions == []


def test_suggest_missing_tags_heuristic():
    from semantic_tags.graph import TagGraph
    from semantic_tags.rag import suggest_missing_tags

    tg = TagGraph()
    tg.graph.add_node(
        "nugget_0", type="nugget", text="I love pizza and pasta", cluster=0, source="a.md"
    )
    tg.graph.add_node("tag_recipe", type="tag", count=1)
    tg.graph.add_edge("nugget_0", "tag_recipe")

    suggestions = suggest_missing_tags(tg)
    assert isinstance(suggestions, list) and suggestions


def test_train_tag_classifier():
    from semantic_tags.graph import TagGraph
    from semantic_tags.classifier import train_tag_classifier

    tg = TagGraph()
    tg.graph.add_node(
        "nugget_0", type="nugget", text="I love this recipe", cluster=0, source="a.md"
    )
    tg.graph.add_node("tag_recipe", type="tag", count=1)
    tg.graph.add_edge("nugget_0", "tag_recipe")

    result = train_tag_classifier(tg)
    assert result is not None
