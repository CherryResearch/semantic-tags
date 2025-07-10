from semantic_tags.diarization import diarize_and_chunk, detect_emotion


def test_diarize_and_chunk():
    text = "Alice: I love this recipe\nBob: I hate cooking"
    chunks = diarize_and_chunk(text)
    assert chunks == [("I love this recipe", "Alice"), ("I hate cooking", "Bob")]


def test_detect_emotion():
    assert detect_emotion("I love it") == "positive"
    assert detect_emotion("I hate it") == "negative"
    assert detect_emotion("just okay") == "neutral"
