"""Hybrid retrieval for RAG: BM25 keyword search + embedding search, fused by reciprocal rank.

Run:  python -m examples.rag_retriever

The embedding function below is a stand-in (hashed word vectors) so the example runs
without a model. Replace `embed` with your model (sentence-transformers, OpenAI, ...);
everything else stays the same.
"""

import zlib

import numpy as np

from pyrust_engine.ai import BM25, chunk_text, cosine_top_k, tokenize

DOCUMENTS = [
    'PyRust compiles Python functions to Rust with one decorator and keeps exact Python results.',
    'Retrieval-augmented generation looks up relevant passages and gives them to a language model.',
    'BM25 ranks documents by how often query words appear, adjusted for document length.',
    'Embeddings map text to vectors; cosine similarity finds passages with similar meaning.',
    'Reciprocal rank fusion combines several ranked lists into one, rewarding items ranked high anywhere.',
    'Chunking splits long documents into overlapping pieces so each fits in the model context.',
    'Python loops are slow because every operation is interpreted; compiled code removes that overhead.',
]


def embed(texts, dim=256):
    """Stand-in embedding: hashed bag of words, normalized. Use a real model in practice."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            out[row, zlib.crc32(token.encode()) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


class HybridRetriever:
    def __init__(self, documents, chunk_size=300, overlap=50, embed=embed):
        self.embed = embed
        self.chunks = [c for d in documents for c in chunk_text(d, chunk_size, overlap)]
        self.bm25 = BM25(self.chunks)
        self.vectors = embed(self.chunks)

    def search(self, query, k=3, depth=20, rrf_k=60):
        keyword_ids, _ = self.bm25.search(query, depth)
        vector_ids, _ = cosine_top_k(self.embed([query])[0], self.vectors, depth)
        fused = {}
        for ranking in (keyword_ids, vector_ids):
            for rank, chunk_id in enumerate(ranking.tolist()):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [(self.chunks[i], round(fused[i], 4)) for i in best]


def main():
    retriever = HybridRetriever(DOCUMENTS)
    for query in ('how does BM25 rank documents', 'make python loops fast', 'combine ranked lists'):
        print(f'\n{query!r}')
        for text, score in retriever.search(query):
            print(f'  {score:.4f}  {text}')


if __name__ == '__main__':
    main()
