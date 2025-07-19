from pathlib import Path
from typing import List, Tuple, Union


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


def load_files(path: Path) -> List[Tuple[Union[str, Path], Path, bool]]:
    """Load text and image files from ``path``.

    Returns a list of tuples ``(content_or_path, relative_path, is_image)``.
    ``content_or_path`` is either the text content or a ``Path`` to the image.
    Supported text files: ``.md``, ``.json``, ``.txt``.
    Image files: ``.jpg``, ``.jpeg``, ``.png``, ``.webp``, ``.gif``.
    """
    items: List[Tuple[Union[str, Path], Path, bool]] = []
    text_exts = {".md", ".json", ".txt"}
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            suf = p.suffix.lower()
            if suf in text_exts:
                items.append((p.read_text(), p.relative_to(path), False))
            elif suf in img_exts:
                items.append((p, p.relative_to(path), True))
    else:
        suf = path.suffix.lower()
        if suf in text_exts:
            items.append((path.read_text(), Path(path.name), False))
        elif suf in img_exts:
            items.append((path, Path(path.name), True))
    return items
