from typing import List, Optional, Union

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image

from .config import DEFAULT_CONFIG, download_model, select_model


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_CONFIG["default_model"],
        batch_size: int = 32,
        device: Optional[str] = None,
        model_dir: Optional[Path] = None,
    ):
        model_name = select_model(model_name)
        if model_dir:
            local_path = Path(model_dir) / model_name.replace("/", "_")
            if local_path.exists():
                model_name = str(local_path)
        self.model = SentenceTransformer(model_name, device=device, cache_folder=str(model_dir) if model_dir else None)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )


class VisionEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_CONFIG["default_vision_model"],
        batch_size: int = 16,
        device: Optional[str] = None,
        model_dir: Optional[Path] = None,
    ):
        model_name = select_model(model_name)
        if model_dir:
            local_path = Path(model_dir) / model_name.replace("/", "_")
            if local_path.exists():
                model_name = str(local_path)
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(model_dir) if model_dir else None,
        )
        self.batch_size = batch_size

    def embed(self, images: List[Union[Path, Image.Image]]) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") if isinstance(p, Path) else p for p in images]
        return self.model.encode(imgs, batch_size=self.batch_size, show_progress_bar=True)

