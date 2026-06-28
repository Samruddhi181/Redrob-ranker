from __future__ import annotations

import re
from dataclasses import dataclass

# Phrases ordered longest → shortest so multi-word matches are found before
# the shorter words they contain ("recommendation engine" before "recommendation").
DOMAIN_SIGNALS: dict[str, list[str]] = {
    "recommendation": [
        "candidate recommendation",
        "job recommendation",
        "item recommendation",
        "product recommendation",
        "collaborative filtering",
        "matrix factorization",
        "two-tower model",
        "two tower model",
        "recommendation engine",
        "recommendation system",
        "recommender system",
        "recommendation model",
        "recommendation pipeline",
        "recommendation",
        "recommender",
        "personalization engine",
        "personalization",
    ],
    "ranking": [
        "learning to rank",
        "pointwise ranking",
        "pairwise ranking",
        "listwise ranking",
        "relevance ranking",
        "ranking pipeline",
        "ranking system",
        "ranking model",
        "re-ranking",
        "reranking",
        "ltr",
        "ranking",
        "ranker",
    ],
    "retrieval": [
        "information retrieval",
        "dense retrieval",
        "sparse retrieval",
        "hybrid retrieval",
        "semantic retrieval",
        "approximate nearest neighbor",
        "vector search",
        "ann search",
        "faiss",
        "qdrant",
        "pinecone",
        "weaviate",
        "milvus",
        "chromadb",
        "chroma",
        "pgvector",
        "elasticsearch",
        "opensearch",
        "retrieval",
    ],
    "search": [
        "query understanding",
        "query expansion",
        "query rewriting",
        "query relaxation",
        "search quality",
        "search relevance",
        "search infrastructure",
        "search platform",
        "search engine",
        "full-text search",
        "semantic search",
        "lexical search",
        "search",
    ],
    "relevance": [
        "relevance judgment",
        "relevance label",
        "relevance annotation",
        "relevance scoring",
        "relevance model",
        "relevance feedback",
        "relevance matching",
        "click-through rate",
        "click through rate",
        "engagement signal",
        "implicit feedback",
        "explicit feedback",
        "dwell time",
        "ctr",
        "relevance",
    ],
    "ab_testing": [
        "experimentation platform",
        "experiment framework",
        "a/b testing",
        "ab testing",
        "a/b test",
        "ab test",
        "online experiment",
        "online evaluation",
        "holdout test",
        "split test",
        "interleaving",
        "bandit",
    ],
    "evaluation": [
        "offline evaluation",
        "evaluation pipeline",
        "ground truth",
        "relevance annotation",
        "mean average precision",
        "human evaluation",
        "evaluation metric",
        "evaluation framework",
        "ndcg",
        "mrr",
        "map@",
        "precision@",
        "recall@",
        "f1@",
    ],
}

# career_description set to 0.70 (not 1.0) so the builder-verb boost is needed
# to reach full credit: "built a ranking system" → 0.70 × 1.45 ≈ 1.0,
# bare mention → 0.70, hedged claim → capped at 0.20/0.15.
SOURCE_WEIGHT: dict[str, float] = {
    "career_description": 0.70,
    "summary":            0.55,
    "headline":           0.35,
}

BUILDER_VERBS: frozenset[str] = frozenset({
    "built", "build", "designed", "design", "developed", "develop",
    "implemented", "implement", "shipped", "ship", "launched", "launch",
    "architected", "engineer", "engineered", "owned", "own",
    "led", "lead", "drove", "drive", "created", "create",
    "improved", "improve", "optimized", "optimize",
    "scaled", "scale", "delivered", "deliver",
    "deployed", "deploy", "established", "establish",
})

EXPOSURE_PHRASES_PASSIVE: tuple[str, ...] = (
    "participated in",
    "supported",
    "assisted",
    "helped",
    "contributed to",
    "part of the team",
    "collaborated on",
)

EXPOSURE_PHRASES_FAMILIAR: tuple[str, ...] = (
    "familiar with",
    "familiarity with",
    "knowledge of",
    "exposure to",
    "understanding of",
    "aware of",
    "experience with",
    "worked with",
    "used",
    "learning",
    "studying",
    "coursework",
    "academic",
    "theoretical",
    "basic knowledge",
    "some experience",
)

_EXPOSURE_STRENGTH_FAMILIAR = 0.20
_EXPOSURE_STRENGTH_PASSIVE  = 0.15
_VERB_LOOKBACK_CHARS = 80
_BUILDER_BOOST = 1.45


@dataclass
class DomainHit:
    domain: str
    strength: float


@dataclass
class EvidenceResult:
    score: float
    domains_found: list[str]


def _has_builder_verb(text_lower: str, signal_start: int) -> bool:
    prefix = text_lower[max(0, signal_start - _VERB_LOOKBACK_CHARS):signal_start]
    words = re.findall(r"[a-z]+", prefix)
    return any(w in BUILDER_VERBS for w in words)


def _exposure_strength(text_lower: str, signal_start: int) -> float | None:
    # Exposure check overrides builder verbs: "familiar with the ranking system I built"
    # is still an awareness claim, not an ownership claim.
    prefix = text_lower[max(0, signal_start - _VERB_LOOKBACK_CHARS):signal_start]
    if any(p in prefix for p in EXPOSURE_PHRASES_PASSIVE):
        return _EXPOSURE_STRENGTH_PASSIVE
    if any(p in prefix for p in EXPOSURE_PHRASES_FAMILIAR):
        return _EXPOSURE_STRENGTH_FAMILIAR
    return None


def _scan_source(text: str, source: str) -> list[DomainHit]:
    if not text:
        return []
    lower = text.lower()
    hits: list[DomainHit] = []
    base_weight = SOURCE_WEIGHT[source]

    for domain, signals in DOMAIN_SIGNALS.items():
        best: DomainHit | None = None
        for signal in signals:  # longest → shortest, so most specific match wins
            pos = lower.find(signal)
            if pos == -1:
                continue
            exp = _exposure_strength(lower, pos)
            if exp is not None:
                strength = exp
            else:
                has_verb = _has_builder_verb(lower, pos)
                strength = min(1.0, base_weight * (_BUILDER_BOOST if has_verb else 1.0))
            hit = DomainHit(domain=domain, strength=strength)
            if best is None or strength > best.strength:
                best = hit
        if best is not None:
            hits.append(best)

    return hits


def extract_evidence(candidate: dict) -> EvidenceResult:
    profile = candidate.get("profile", {})
    all_hits: list[DomainHit] = []

    # Scan each role separately so a verb in one role can't pair with a term from another.
    for role in candidate.get("career_history", []):
        all_hits.extend(_scan_source(role.get("description", ""), "career_description"))

    all_hits.extend(_scan_source(profile.get("summary", ""), "summary"))
    all_hits.extend(_scan_source(profile.get("headline", ""), "headline"))

    best: dict[str, float] = {}
    for hit in all_hits:
        if hit.strength > best.get(hit.domain, 0.0):
            best[hit.domain] = hit.strength

    score = sum(best.values()) / len(DOMAIN_SIGNALS)
    domains_found = sorted(best, key=lambda d: -best[d])

    return EvidenceResult(score=min(1.0, score), domains_found=domains_found)
