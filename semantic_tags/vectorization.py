from typing import List, Optional

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

