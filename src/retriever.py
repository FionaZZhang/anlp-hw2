"""
Retrieval module with dense, sparse, and hybrid retrieval.
"""
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import bm25s
import Stemmer


class DenseRetriever:
    """Dense retrieval using sentence transformers and FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = None

    def index_documents(self, documents: List[Dict]):
        """Create FAISS index from documents."""
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        print("Encoding documents for dense retrieval...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) 
        self.index.add(embeddings.astype('float32'))

        print(f"Indexed {len(documents)} documents in FAISS")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve top-k documents for a query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results


class SparseRetriever:
    """Sparse retrieval using BM25."""

    def __init__(self):
        self.retriever = None
        self.documents = None
        self.stemmer = Stemmer.Stemmer("english")

    def index_documents(self, documents: List[Dict]):
        """Create BM25 index from documents."""
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        print("Building BM25 index...")
        # Tokenize and stem
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemmer)

        # Create BM25 retriever
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        print(f"Indexed {len(documents)} documents with BM25")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve top-k documents for a query."""
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        output = []
        for idx, score in zip(results[0], scores[0]):
            if idx < len(self.documents):
                output.append((self.documents[idx], float(score)))

        return output


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods using Reciprocal Rank Fusion."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.dense = DenseRetriever(model_name)
        self.sparse = SparseRetriever()
        self.documents = None

    def index_documents(self, documents: List[Dict]):
        """Index documents for both retrievers."""
        self.documents = documents
        self.dense.index_documents(documents)
        self.sparse.index_documents(documents)

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict, float]],
        sparse_results: List[Tuple[Dict, float]],
        k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        RRF score = sum(1 / (k + rank)) for each result list
        """
        scores = {}
        doc_map = {}

        # Process dense results
        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        # Process sparse results
        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = doc['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [(doc_map[doc_id], scores[doc_id]) for doc_id in sorted_ids]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        method: str = "rrf"
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve documents using hybrid approach.

        Args:
            query: Search query
            top_k: Number of results to return
            dense_weight: Weight for dense retrieval (only used in weighted method)
            method: Fusion method - "rrf" or "weighted"
        """
        n_candidates = top_k * 3

        dense_results = self.dense.retrieve(query, n_candidates)
        sparse_results = self.sparse.retrieve(query, n_candidates)

        if method == "rrf":
            combined = self.reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            scores = {}
            doc_map = {}

            if dense_results:
                max_dense = max(s for _, s in dense_results) or 1
                for doc, score in dense_results:
                    doc_id = doc['id']
                    normalized_score = score / max_dense
                    scores[doc_id] = scores.get(doc_id, 0) + dense_weight * normalized_score
                    doc_map[doc_id] = doc

            if sparse_results:
                max_sparse = max(s for _, s in sparse_results) or 1
                for doc, score in sparse_results:
                    doc_id = doc['id']
                    normalized_score = score / max_sparse
                    scores[doc_id] = scores.get(doc_id, 0) + (1 - dense_weight) * normalized_score
                    doc_map[doc_id] = doc

            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            combined = [(doc_map[doc_id], scores[doc_id]) for doc_id in sorted_ids]

        return combined[:top_k]


def load_retriever(documents_path: str, retriever_type: str = "hybrid") -> HybridRetriever:
    """Load documents and initialize retriever."""
    with open(documents_path, 'r') as f:
        documents = json.load(f)

    if retriever_type == "dense":
        retriever = DenseRetriever()
    elif retriever_type == "sparse":
        retriever = SparseRetriever()
    else:
        retriever = HybridRetriever()

    retriever.index_documents(documents)
    return retriever


if __name__ == "__main__":
    retriever = load_retriever("data/processed/documents.json")

    test_query = "When was Carnegie Mellon University founded?"
    results = retriever.retrieve(test_query, top_k=3)

    print(f"\nQuery: {test_query}")
    print("\nTop results:")
    for doc, score in results:
        print(f"  [{score:.4f}] {doc['title'][:50]}...")
        print(f"    {doc['text'][:200]}...")
