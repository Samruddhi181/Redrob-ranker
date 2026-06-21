# Lightweight, deterministic TF-IDF + cosine similarity.
#
# Used to score how closely a candidate's profile text overlaps with the job
# description's full text. This replaces the earlier approach of counting
# exact hits against a fixed buzzword list: instead of asking "does this
# literal word appear," it asks "how much does this candidate's vocabulary,
# weighted by how distinctive each word is across the whole candidate pool,
# look like the JD's vocabulary." Two profiles that both say "ranking" no
# longer score identically regardless of context; profiles that share rare,
# JD-specific words (e.g. "qdrant", "ndcg") score higher than profiles that
# only share common words like "team" or "engineer."
#
# No external libraries (no numpy/sklearn) so this stays dependency-free,
# fully deterministic, and auditable line by line.
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z][a-z0-9+/.\-]*")

# A short, generic stopword list -- common English function words that would
# otherwise dominate every document's term frequency and drown out the
# distinctive, JD-relevant vocabulary TF-IDF is meant to surface.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "had", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "over", "such", "than", "that", "the",
    "their", "then", "these", "this", "those", "to", "under", "was", "we",
    "were", "will", "with", "you", "your", "they", "i", "not", "also",
    "about", "more", "most", "across", "using", "use", "used",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; stopwords and pure punctuation dropped.

    Keeps internal hyphens/slashes/dots so multi-part terms like
    "fine-tuning", "a/b", or "ndcg@10"-style tokens survive as one token
    rather than being split into noise.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


class IDFModel:
    """Inverse-document-frequency weights learned from a document corpus.

    Uses "smooth" IDF -- the same formula scikit-learn's TfidfVectorizer
    uses by default: idf(t) = ln((1 + n) / (1 + df(t))) + 1. Always
    positive, well-defined even for a term that appears in every document,
    and gives an explicit, sane default weight for a term that never
    appeared in the corpus at all (treated the same as df(t) = 0 -- the
    most distinctive possible term).
    """

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
    """One streaming pass over the corpus: for each term, count how many
    documents it appears in at least once (document frequency). Takes an
    iterable of already-tokenized documents so the caller controls how
    each document's text is collected and tokenized.
    """
    doc_freq: dict[str, int] = {}
    doc_count = 0
    for tokens in token_lists:
        doc_count += 1
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return IDFModel(doc_count, doc_freq)


def tfidf_vector(tokens: list[str], idf: IDFModel) -> dict[str, float]:
    """Sparse TF-IDF vector: raw term count * corpus IDF weight.

    Raw count (rather than a length-normalized count) is fine here --
    cosine similarity divides by each vector's own norm, so any uniform
    per-document scaling factor cancels out exactly and doesn't change the
    result.
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {term: count * idf.weight(term) for term, count in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors, in [0, 1] for the
    non-negative TF-IDF weights produced above."""
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
