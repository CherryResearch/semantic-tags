import sys
import types

# Stub heavy dependencies similarly to test_pipeline
sys.modules["numpy"] = types.ModuleType("numpy")
sys.modules["numpy"].ndarray = list
sys.modules["numpy"].random = types.SimpleNamespace(rand=lambda *a, **k: [[0] * 2 for _ in range(a[0])])

st_module = types.ModuleType("sentence_transformers")
st_module.SentenceTransformer = lambda *a, **k: types.SimpleNamespace(
    encode=lambda texts, batch_size=32, show_progress_bar=False: [[0] * 2 for _ in texts],
    device=k.get("device", "cpu"),
    model_name=a[0] if a else "model",
)
sys.modules["sentence_transformers"] = st_module

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

weaviate_mod = types.ModuleType("weaviate")
weaviate_mod.Client = lambda *a, **k: types.SimpleNamespace(
    schema=types.SimpleNamespace(get=lambda: {"classes": []}),
    data_object=types.SimpleNamespace(create=lambda *args, **kwargs: None),
)
sys.modules["weaviate"] = weaviate_mod

from semantic_tags.config import DEFAULT_CONFIG_PATH
from semantic_tags.cli import main

def test_cli_show_config(capsys):
    sys.argv = ["prog", "--show-config"]
    main()
    captured = capsys.readouterr()
    assert str(DEFAULT_CONFIG_PATH) in captured.out
    assert "model_dir" in captured.out
    assert "batch_size" in captured.out
    assert "device" in captured.out
