from semantic_tags.topic_inference import infer_cluster_tags


def test_infer_cluster_tags():
    nuggets = ["I like tasty recipes", "This recipe is easy", "Anime is cool"]
    labels = [0, 0, 1]
    topics = infer_cluster_tags(nuggets, labels)
    assert 0 in topics and 1 in topics
    assert isinstance(topics[0], str)
