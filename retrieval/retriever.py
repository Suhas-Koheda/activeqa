"""
BM25 retriever built with rank_bm25.
Corpus is loaded from the dataset documents.
"""
from __future__ import annotations
import os
import pickle
from rank_bm25 import BM25Okapi
from config import DATA_DIR, INDEX_PATH


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(documents: dict[str, str]) -> BM25Okapi:
    """
    Build BM25 index from a {doc_id: text} mapping.
    Returns a BM25Okapi object.
    """
    doc_ids   = list(documents.keys())
    tokenized = [tokenize(documents[did]) for did in doc_ids]
    bm25      = BM25Okapi(tokenized)
    return bm25, doc_ids


class BM25Retriever:
    def __init__(self, documents: dict[str, str], index_path: str | None = None):
        self.documents  = documents
        self.index_path = index_path or os.path.join(DATA_DIR, "bm25_index.pkl")

        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                saved = pickle.load(f)
            self.bm25   = saved["bm25"]
            self.doc_ids = saved["doc_ids"]
        else:
            self.bm25, self.doc_ids = build_bm25_index(documents)
            self.save_index()

    def save_index(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return top-k doc_ids for a query."""
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.doc_ids[i] for i in top_indices]
