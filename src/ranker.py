from __future__ import annotations

from typing import Iterable
import heapq

from .candidate_io import read_docx_text, read_jsonl
from .features import score_candidate
from .jd_profile import extract_jd_profile
from .reasoning import build_reasoning


def rank_candidates(jd_text: str, candidates: Iterable[dict]) -> list[dict]:
    jd = extract_jd_profile(jd_text)
    heap: list[tuple[float, int, dict]] = []
    for candidate in candidates:
        features = score_candidate(candidate, jd.years_min, jd.years_max, jd.must_have_terms)
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
