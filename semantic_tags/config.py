from __future__ import annotations

from pathlib import Path
import json
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "model_config.json"

DEFAULT_CONFIG = {
    "model_dir": str(REPO_ROOT / "models"),
    "default_model": "sentence-transformers/all-mpnet-base-v2",
}

AVAILABLE_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "distilroberta": "sentence-transformers/all-distilroberta-v1",
    "multiqa-mpnet": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "multiqa-minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
}


def load_config(path: Path | None = None) -> dict:
    config_path = Path(os.getenv("SEMANTIC_TAGS_CONFIG", path or DEFAULT_CONFIG_PATH))
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(data)
    return cfg


def save_config(config: dict, path: Path | None = None) -> None:
    config_path = Path(os.getenv("SEMANTIC_TAGS_CONFIG", path or DEFAULT_CONFIG_PATH))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def download_model(model_name: str, model_dir: Path) -> Path:
    """Download a sentence-transformer model to the given directory."""
    from sentence_transformers import SentenceTransformer

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / model_name.replace("/", "_")
    if not target.exists():
        st_model = SentenceTransformer(model_name)
        st_model.save(str(target))
    return target


def select_model(name: str) -> str:
    """Return the full model name for a known alias."""
    return AVAILABLE_MODELS.get(name, name)

