from sentence_transformers import CrossEncoder


class BGEReranker:
    """
    BGE cross-encoder reranker
    """

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=5):
        if not chunks:
            return []

        pairs = [(query, c["content"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [c for c, _ in ranked[:top_k]]