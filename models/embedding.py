"""BGE 句子嵌入模型封装。"""

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f" Embedding 加载成功 | 设备: {self.device}")

    def get_embeddings(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
