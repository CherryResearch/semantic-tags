from pathlib import Path
from typing import List, Tuple


def load_transcripts(path: Path) -> List[Tuple[str, Path]]:
    """Load transcripts from ``path`` and return their contents with relative paths.

    The path can be a single file or a directory containing Markdown or JSON
    files.  Each returned tuple consists of the file text and the file path
    relative to ``path`` when ``path`` is a directory, or just the file name
    when ``path`` is a file.
    """
    texts: List[Tuple[str, Path]] = []
    if path.is_dir():
        for p in sorted(path.rglob("*.md")):
            texts.append((p.read_text(), p.relative_to(path)))
        for p in sorted(path.rglob("*.json")):
            texts.append((p.read_text(), p.relative_to(path)))
    else:
        texts.append((path.read_text(), Path(path.name)))
    return texts
