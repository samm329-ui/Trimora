from typing import Optional


class PatternEmbeddings:
    def __init__(self):
        self._embeddings: dict[str, list[float]] = {}
        self._dimension: int = 0

    def add_embedding(self, pattern_key: str, embedding: list[float]) -> None:
        self._embeddings[pattern_key] = embedding
        self._dimension = max(self._dimension, len(embedding))

    def get_embedding(self, pattern_key: str) -> Optional[list[float]]:
        return self._embeddings.get(pattern_key)

    def find_similar(self, query_embedding: list[float],
                     top_n: int = 5) -> list[dict]:
        if not self._embeddings:
            return []

        scores = []
        for key, emb in self._embeddings.items():
            sim = self._cosine_similarity(query_embedding, emb)
            scores.append({"pattern_key": key, "similarity": sim})

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_n]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(ai * bi for ai, bi in zip(a, b))
        mag_a = sum(ai ** 2 for ai in a) ** 0.5
        mag_b = sum(bi ** 2 for bi in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def embedding_count(self) -> int:
        return len(self._embeddings)

    def clear(self) -> None:
        self._embeddings.clear()
        self._dimension = 0
