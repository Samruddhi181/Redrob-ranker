from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z][a-z0-9+/.\-]*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "had", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "over", "such", "than", "that", "the",
    "their", "then", "these", "this", "those", "to", "under", "was", "we",
    "were", "will", "with", "you", "your", "they", "i", "not", "also",
    "about", "more", "most", "across", "using", "use", "used",
}

# Maps specific tool names and phrases to canonical concept tokens, applied to
# both the candidate profile and the JD so that e.g. "FAISS" on a profile and
# "vector databases" in the JD both emit "concept.vectordb" and meet in cosine
# similarity even though the literal words differ.
CONCEPT_EXPANSIONS: dict[str, str] = {
    "vector databases":       "concept.vectordb",
    "vector database":        "concept.vectordb",
    "vector store":           "concept.vectordb",
    "vector search":          "concept.vectordb",
    "ann index":              "concept.vectordb",
    "faiss":                  "concept.vectordb",
    "qdrant":                 "concept.vectordb",
    "pinecone":               "concept.vectordb",
    "weaviate":               "concept.vectordb",
    "milvus":                 "concept.vectordb",
    "chroma":                 "concept.vectordb",
    "chromadb":               "concept.vectordb",
    "pgvector":               "concept.vectordb",
    "vespa":                  "concept.vectordb",
    "information retrieval":  "concept.retrieval",
    "dense retrieval":        "concept.retrieval",
    "hybrid retrieval":       "concept.retrieval",
    "hybrid search":          "concept.retrieval",
    "semantic search":        "concept.retrieval",
    "lexical search":         "concept.retrieval",
    "retrieval":              "concept.retrieval",
    "sentence-transformers":  "concept.embeddings",
    "sentence transformers":  "concept.embeddings",
    "text embeddings":        "concept.embeddings",
    "openai embeddings":      "concept.embeddings",
    "embeddings":             "concept.embeddings",
    "embedding":              "concept.embeddings",
    "word2vec":               "concept.embeddings",
    "bge":                    "concept.embeddings",
    "learning to rank":       "concept.ranking",
    "re-ranking":             "concept.ranking",
    "reranking":              "concept.ranking",
    "relevance ranking":      "concept.ranking",
    "ranking":                "concept.ranking",
    "ranker":                 "concept.ranking",
    "bm25":                   "concept.ranking",
    "mean average precision": "concept.rankeval",
    "ndcg":                   "concept.rankeval",
    "mrr":                    "concept.rankeval",
    "collaborative filtering": "concept.recsys",
    "recommender systems":     "concept.recsys",
    "recommender system":      "concept.recsys",
    "recommendation systems":  "concept.recsys",
    "recommendation system":   "concept.recsys",
    "recommendation":          "concept.recsys",
    "recommender":             "concept.recsys",
    "fine-tuning":             "concept.finetuning",
    "fine tuning":             "concept.finetuning",
    "qlora":                   "concept.finetuning",
    "lora":                    "concept.finetuning",
    "peft":                    "concept.finetuning",
    "rlhf":                    "concept.finetuning",
}


def concept_expand(text: str) -> str:
    lower = text.lower()
    added: set[str] = set()
    for phrase, concept in CONCEPT_EXPANSIONS.items():
        if phrase in lower:
            added.add(concept)
    if not added:
        return text
    return text + " " + " ".join(sorted(added))


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


class IDFModel:
    # Smooth IDF: ln((1+n)/(1+df)) + 1 — same formula as sklearn's TfidfVectorizer default.
    # Always positive; unknown terms get the maximum weight (df=0).

    __slots__ = ("document_count", "_weights", "_default_weight")

    def __init__(self, document_count: int, doc_freq: dict[str, int]):
        self.document_count = document_count
        self._weights = {
            term: math.log((1 + document_count) / (1 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        self._default_weight = math.log((1 + document_count) / 1) + 1.0

    def weight(self, term: str) -> float:
        return self._weights.get(term, self._default_weight)


def build_idf(token_lists: Iterable[list[str]]) -> IDFModel:
    doc_freq: dict[str, int] = {}
    doc_count = 0
    for tokens in token_lists:
        doc_count += 1
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return IDFModel(doc_count, doc_freq)


def tfidf_vector(tokens: list[str], idf: IDFModel) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {term: count * idf.weight(term) for term, count in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(weight * vec_b[term] for term, weight in vec_a.items() if term in vec_b)
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
