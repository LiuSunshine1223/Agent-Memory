"""评测脚本自检用的替身模型。

这些替身只用来确认评测流程本身能跑通、打分逻辑正确，
不代表系统的真实表现。正式结果必须用真实嵌入模型和真实 LLM 运行。
"""

from __future__ import annotations

import hashlib

import numpy as np


class HashEmbedder:
    """确定性的字符 n-gram 哈希向量，接口与 models.embedding.EmbeddingModel 相同。

    它没有语义能力：语义检索类指标在替身模式下没有意义，报告里会标出来。
    """

    def __init__(self, dimension: int = 512):
        self.dimension = int(dimension)

    def get_embeddings(self, texts):
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            normalized = "".join(str(text).lower().split())
            grams = [normalized[i:i + n] for n in (1, 2, 3) for i in range(len(normalized) - n + 1)]
            for gram in grams:
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class StubChatLLM:
    """对话用替身：立即返回固定回答，用于检查延迟评测流程。"""

    def ask(self, prompt, *, system=None):
        return "（替身模型回答）"

    def ask_json(self, prompt, *, schema=None, system=None):
        return '{"is_user_fact": false, "memories": []}'
