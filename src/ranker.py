from __future__ import annotations

from datetime import date
from typing import Callable, Iterable
import heapq

from .features import (
    candidate_profile_text,
    extract_demand_raw,
    extract_last_active_date,
    learn_demand_caps,
    score_candidate,
)
from .jd_profile import extract_jd_profile
from .reasoning import build_reasoning
from .tfidf import build_idf, concept_expand, tfidf_vector, tokenize


def rank_candidates(jd_text: str, candidates_factory: Callable[[], Iterable[dict]]) -> list[dict]:
    # Factory instead of a single iterator: TF-IDF needs two passes over the pool
    # (IDF weights, then scoring), so the caller must be able to reopen the source.
    jd = extract_jd_profile(jd_text)

    # Anchored to pool's own max last_active_date, not wall-clock, so output stays
    # deterministic regardless of what day the script runs.
    dataset_as_of: list[date | None] = [None]
    demand_raw: list[tuple[float, float, float]] = []

    def _profile_tokens_tracking_activity() -> Iterable[list[str]]:
        for c in candidates_factory():
            last_active = extract_last_active_date(c)
            if last_active is not None and (dataset_as_of[0] is None or last_active > dataset_as_of[0]):
                dataset_as_of[0] = last_active
            demand_raw.append(extract_demand_raw(c))
            yield tokenize(concept_expand(candidate_profile_text(c)))

    idf = build_idf(_profile_tokens_tracking_activity())
    jd_vector = tfidf_vector(tokenize(concept_expand(jd_text)), idf)
    as_of = dataset_as_of[0] or date.today()
    demand_caps = learn_demand_caps(demand_raw)

    heap: list[tuple[float, int, dict]] = []
    for candidate in candidates_factory():
        features = score_candidate(candidate, jd.years_min, jd.years_max, idf, jd_vector, as_of, demand_caps)
        reasoning = build_reasoning(candidate, features)
        rounded_score = round(features.final_score, 4)
        candidate_number = int(candidate["candidate_id"].split("_")[1])
        row = (
            rounded_score,
            -candidate_number,
            {
                "candidate_id": candidate["candidate_id"],
                "rank": 0,
                "score": rounded_score,
                "reasoning": reasoning,
                "_features": features,
            },
        )
        if len(heap) < 100:
            heapq.heappush(heap, row)
        else:
            heapq.heappushpop(heap, row)

    results = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[2]["candidate_id"]))]
    for index, row in enumerate(results, start=1):
        row["rank"] = index
    return results
