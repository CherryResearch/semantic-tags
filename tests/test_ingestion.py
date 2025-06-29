import os
from pathlib import Path

from semantic_tags.ingestion import load_transcripts


def test_load_transcripts_nested(tmp_path):
    # create files in nested directories
    (tmp_path / "root.md").write_text("root")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inner.md").write_text("inner")
    nested = sub / "nested"
    nested.mkdir()
    (nested / "deep.json").write_text("{}")

    results = load_transcripts(tmp_path)
    paths = sorted(p for _, p in results)
    assert paths == sorted([
        Path("root.md"),
        Path("sub/inner.md"),
        Path("sub/nested/deep.json"),
    ])
    assert len(results) == 3
