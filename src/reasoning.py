from __future__ import annotations

from typing import Any


def _collect_profile_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for role in candidate.get("career_history") or []:
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for skill in candidate.get("skills") or []:
        parts.append(skill.get("name", ""))
    return " ".join(parts).lower()


def _domain_phrase(candidate: dict[str, Any]) -> str:
    text = _collect_profile_text(candidate)
    if "recommendation" in text and ("retrieval" in text or "ranking" in text):
        return "recommendation and retrieval systems"
    if "ranking" in text and "retrieval" in text:
        return "information retrieval and ranking systems"
    if "search" in text and ("ranking" in text or "retrieval" in text):
        return "search and ranking systems"
    if "recommendation" in text:
        return "recommendation systems"
    if "retrieval" in text:
        return "information retrieval systems"
    if "ranking" in text:
        return "ranking systems"
    if "search" in text:
        return "search systems"
    if "embedding" in text or "embeddings" in text:
        return "embedding-based ML systems"
    if "llm" in text or "fine-tuning" in text:
        return "large language model systems"
    return "applied ML and AI systems"


def _opening_sentence(candidate: dict[str, Any], features) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title") or "Engineer"
    years = features.years_of_experience
    domain = _domain_phrase(candidate)
    return f"{title} with {years:.1f} years of experience in {domain}."


_DOMAIN_LABELS: dict[str, str] = {
    "recommendation": "recommendation",
    "ranking":        "ranking",
    "retrieval":      "retrieval",
    "search":         "search",
    "relevance":      "relevance matching",
    "ab_testing":     "A/B testing",
    "evaluation":     "offline evaluation",
}


def _evidence_domains(evidence: list[str]) -> str:
    seen: set[str] = set()
    found: list[str] = []
    for d in evidence:
        label = _DOMAIN_LABELS.get(d)
        if label and label not in seen:
            seen.add(label)
            found.append(label)
    if not found:
        return ""
    if len(found) == 1:
        return found[0]
    return ", ".join(found[:-1]) + " and " + found[-1]


def _evidence_and_behavioral_sentence(candidate: dict[str, Any], features) -> str:
    evidence = features.matched_evidence
    signals = candidate.get("redrob_signals", {})

    response_rate = float(signals.get("recruiter_response_rate") or 0.0) * 100
    open_to_work = bool(signals.get("open_to_work_flag"))
    last_active = signals.get("last_active_date")

    domains = _evidence_domains(evidence)
    if domains:
        ev_part = f"Career history includes work on {domains}-related systems"
    else:
        ev_part = "Career history includes relevant technical experience"

    if response_rate >= 70 and open_to_work:
        beh = "the candidate maintains a strong recruiter response rate and recent platform activity"
    elif response_rate >= 70:
        beh = f"the candidate maintains a strong recruiter response rate ({response_rate:.0f}%)"
    elif open_to_work and last_active:
        beh = f"the candidate is active on the platform as of {last_active} and flagged open to work"
    elif open_to_work:
        beh = "the candidate is flagged open to work"
    elif response_rate >= 40:
        beh = f"recruiter response rate is {response_rate:.0f}%"
    else:
        beh = f"recruiter response rate is low ({response_rate:.0f}%)"

    return f"{ev_part}, and {beh}."


def _concern_sentence(candidate: dict[str, Any], features) -> str | None:
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    if features.disqualifier_concerns:
        c = features.disqualifier_concerns[0]
        return (c[0].upper() + c[1:]).rstrip(".") + "."

    if features.concerns:
        c = features.concerns[0]
        return (c[0].upper() + c[1:]).rstrip(".") + "."

    notice = signals.get("notice_period_days")
    if notice is not None and int(notice) > 60:
        return f"Notice period of {notice} days may slow hiring."

    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    if "india" not in country and "india" not in location:
        if not signals.get("willing_to_relocate"):
            return "Based outside India and not flagged as open to relocation."

    return None


def build_reasoning(candidate: dict[str, Any], features) -> str:
    parts = [
        _opening_sentence(candidate, features),
        _evidence_and_behavioral_sentence(candidate, features),
    ]
    concern = _concern_sentence(candidate, features)
    if concern:
        parts.append(concern)
    return " ".join(parts)
