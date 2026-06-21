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
from .tfidf import build_idf, tfidf_vector, tokenize


def rank_candidates(jd_text: str, candidates_factory: Callable[[], Iterable[dict]]) -> list[dict]:
    """`candidates_factory` is a zero-arg callable that returns a fresh
    iterator over the full candidate pool each time it's called (e.g.
    `lambda: read_jsonl(path)`, which reopens the file). TF-IDF needs two
    passes over the pool -- one to learn the corpus's IDF weights from
    everyone's profile text, one to score everyone against the JD using
    those weights -- so candidates can't be drawn from a single, already-
    started generator. The factory lets each pass start fresh without ever
    holding all 100K candidates in memory at once.
    """
    jd = extract_jd_profile(jd_text)

    # The pool's own most recent last_active_date doubles as the "as of"
    # anchor for everyone's activity_score (features.py) -- tracked as a
    # side effect of the IDF pass we're already streaming, so this stays a
    # two-pass pipeline rather than three, and never depends on wall-clock
    # time (which would make the submitted CSV drift if re-run on a later
    # date, breaking byte-identical reproducibility).
    dataset_as_of: list[date | None] = [None]

    # Recruiter-demand fields (saved_by_recruiters_30d, etc.) are unbounded,
    # right-skewed counts -- a candidate's raw count means little without
    # knowing where it sits relative to the rest of the pool. Collected as
    # another side effect of this same first pass (still no third pass over
    # 100K candidates) and turned into per-field p95 caps once the pass
    # finishes, the same "learn it from the corpus itself" approach already
    # used for IDF weights and dataset_as_of.
    demand_raw: list[tuple[float, float, float]] = []

    def _profile_tokens_tracking_activity() -> Iterable[list[str]]:
        for c in candidates_factory():
            last_active = extract_last_active_date(c)
            if last_active is not None and (dataset_as_of[0] is None or last_active > dataset_as_of[0]):
                dataset_as_of[0] = last_active
            demand_raw.append(extract_demand_raw(c))
            yield tokenize(candidate_profile_text(c))

    idf = build_idf(_profile_tokens_tracking_activity())
    jd_vector = tfidf_vector(tokenize(jd_text), idf)
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
