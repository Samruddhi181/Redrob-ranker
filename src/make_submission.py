from __future__ import annotations

import csv
import sys
from pathlib import Path

from .candidate_io import read_docx_text, read_jsonl
from .ranker import rank_candidates


DEBUG_FIELDNAMES = [
    "candidate_id",
    "rank",
    "score",
    "capability_match",
    "career_evidence_fit",
    "production_experience_fit",
    "behavioral_fit",
    "experience_alignment",
    "logistics_fit",
    "recruiter_demand",
    "honeypot_risk",
    "disqualifier_risk",
    "matched_evidence",
    "concerns",
    "disqualifier_concerns",
    "reasoning",
    "manual_label",
    "manual_notes",
]


def _format_score(value: float) -> str:
    return f"{value:.4f}"


def write_submission_csv(output_path: Path, rows: list[dict]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": _format_score(row["score"]),
                    "reasoning": row["reasoning"],
                }
            )


def write_debug_csv(debug_path: Path, rows: list[dict]) -> None:
    with open(debug_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEBUG_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            features = row["_features"]
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": _format_score(row["score"]),
                    "capability_match": _format_score(features.capability_match),
                    "career_evidence_fit": _format_score(features.career_evidence_fit),
                    "production_experience_fit": _format_score(features.production_experience_fit),
                    "behavioral_fit": _format_score(features.behavioral_fit),
                    "experience_alignment": _format_score(features.experience_alignment),
                    "logistics_fit": _format_score(features.logistics_fit),
                    "recruiter_demand": _format_score(features.recruiter_demand),
                    "honeypot_risk": _format_score(features.honeypot_risk),
                    "disqualifier_risk": _format_score(features.disqualifier_risk),
                    "matched_evidence": "; ".join(features.matched_evidence),
                    "concerns": "; ".join(features.concerns),
                    "disqualifier_concerns": "; ".join(features.disqualifier_concerns),
                    "reasoning": row["reasoning"],
                    "manual_label": "",
                    "manual_notes": "",
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 3:
        print("Usage: python -m src.make_submission <job_description.docx> <candidates.jsonl.gz> <output.csv>")
        return 1

    jd_path = Path(args[0])
    candidates_path = Path(args[1])
    output_path = Path(args[2])

    jd_text = read_docx_text(jd_path)
    top = rank_candidates(jd_text, lambda: read_jsonl(candidates_path))
    debug_path = output_path.with_name("debug_top100.csv")

    write_submission_csv(output_path, top)
    write_debug_csv(debug_path, top)

    print(f"Wrote {output_path}")
    print(f"Wrote {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
