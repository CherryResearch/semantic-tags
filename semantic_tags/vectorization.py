from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32, device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
