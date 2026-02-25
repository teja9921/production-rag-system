import cohere

class CohereReranker:

    def __init__(self, api_key):
        self.client = cohere.Client(api_key)

    def rerank(self, query, chunks, top_k=5):
        if not chunks:
            return []

        docs = [c["content"] for c in chunks]

        response = self.client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_k
        )

        ranked_chunks = [
            chunks[r.index] for r in response.results
        ]

        return ranked_chunks