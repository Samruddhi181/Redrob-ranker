# why did we rank this candidate?
from __future__ import annotations

from typing import Any


def build_reasoning(candidate: dict[str, Any], features) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    years = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "professional")
    location = profile.get("location", "their current location")

    evidence = features.matched_evidence[:2]
    evidence_text = ", ".join(evidence) if evidence else "relevant project and skill evidence"

    reasons = [
        f"{years} years of experience as a {title} with evidence of {evidence_text}.",
    ]

    if signals.get("open_to_work_flag"):
        reasons.append("Recently open to work, which improves shortlist usefulness.")
    if signals.get("preferred_work_mode") in {"hybrid", "flexible"}:
        reasons.append(f"Work-mode preference is compatible with the role's hybrid setup from {location}.")
    if features.concerns:
        reasons.append(f"Some caution remains: {features.concerns[0]}.")

    return " ".join(reasons[:3])

