from rank_bm25 import BM25Okapi
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

class BM25Store:
    """
    Sparse lexical retriever
    """

    def __init__(self, path:str):
        self.path = Path(path)
        self.bm25 = None
        self.documents: List[Dict[str, Any]] =[]
    
    def _preprocess(self, text: str) -> List[str]:
        """
        Cleans text by lowercasing, removing punctuation, and splitting into tokens.
        """
        # 1. Lowercase: "Running!" -> "running!"
        text = text.lower().replace('/', ' ')
        # 2. Remove Punctuation: "running!" -> "running"
        text = re.sub('[^\\w\\s]','', text)
        # 3. Tokenize: "hello world" -> ["hello", "world"]
        return text.split()

    def build(self, chunks: List[Dict[str, Any]]):
        # Apply preprocessing to every document in the corpus
        corpus = [self._preprocess(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.documents = chunks
    
    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump((self.bm25, self.documents), f)

    def load(self):
        with open(self.path, "rb") as f:
            self.bm25, self.documents = pickle.load(f)
    
    def search(self, query: str, k: int):
        # Preprocess the query the EXACT same way as the documents
        tokens = self._preprocess(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse= True)[:k]
        return [(self.documents[i], scores[i]) for i in top_indices]