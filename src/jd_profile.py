# What does the job actually want?
from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class JDProfile:
    raw_text: str
    years_min: float = 0.0
    years_max: float = 50.0
    # Fixed-vocabulary hits, kept for visibility/debugging only. The
    # capability_match score no longer counts these directly -- it uses
    # TF-IDF cosine similarity over the JD's full raw_text instead, which
    # isn't limited to this fixed list (see src/tfidf.py, src/ranker.py).
    must_have_terms: list[str] = field(default_factory=list)
    good_terms: list[str] = field(default_factory=list)
    location_terms: list[str] = field(default_factory=list)
    logistics_terms: list[str] = field(default_factory=list)
    disqualifier_terms: list[str] = field(default_factory=list)


TECH_TERMS = [
    "embedding",
    "embeddings",
    "retrieval",
    "ranking",
    "search",
    "recommendation",
    "vector",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "llm",
    "fine-tuning",
    "fine tuning",
    "python",
    "evaluation",
    "ndcg",
    "map",
    "mrr",
]


def extract_jd_profile(text: str) -> JDProfile:
    lower = text.lower()

    years_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*years", lower)
    years_min, years_max = 0.0, 50.0
    if years_match:
        years_min = float(years_match.group(1))
        years_max = float(years_match.group(2))
    else:
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years", lower)
        if range_match:
            years_min = float(range_match.group(1))
            years_max = float(range_match.group(2))

    must_have = []
    for term in TECH_TERMS:
        if term in lower:
            must_have.append(term)

    location_terms = []
    for term in ["pune", "noida", "india", "hybrid", "relocation", "tier-1"]:
        if term in lower:
            location_terms.append(term)

    logistics_terms = []
    for term in ["notice period", "open to work", "relocate", "remote", "onsite", "hybrid"]:
        if term in lower:
            logistics_terms.append(term)

    disqualifiers = []
    for term in ["consulting", "langchain", "marketing manager", "title-chasers"]:
        if term in lower:
            disqualifiers.append(term)

    return JDProfile(
        raw_text=text,
        years_min=years_min,
        years_max=years_max,
        must_have_terms=must_have,
        good_terms=["production", "evaluation", "ranking", "retrieval", "search", "python", "deploy"],
        location_terms=location_terms,
        logistics_terms=logistics_terms,
        disqualifier_terms=disqualifiers,
    )
