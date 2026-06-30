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


def _most_recent_company(candidate: dict[str, Any]) -> str | None:
    roles = candidate.get("career_history") or []
    if not roles:
        return None
    company = roles[0].get("company", "").strip()
    return company if company else None


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
    company = _most_recent_company(candidate)
    if company:
        return f"{title} with {years:.1f} years of experience in {domain}, most recently at {company}."
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


def _top_domains(evidence: list[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    found: list[str] = []
    for d in evidence:
        label = _DOMAIN_LABELS.get(d)
        if label and label not in seen:
            seen.add(label)
            found.append(label)
        if len(found) == limit:
            break
    return found


def _domains_phrase(domains: list[str]) -> str:
    if not domains:
        return ""
    if len(domains) == 1:
        return domains[0]
    return ", ".join(domains[:-1]) + " and " + domains[-1]


def _evidence_sentence(features) -> str:
    domains = _top_domains(features.matched_evidence)
    if not domains:
        return "Career history shows relevant technical experience."
    phrase = _domains_phrase(domains)
    career_evidence = features.career_evidence_fit
    if career_evidence >= 0.65:
        return f"Has hands-on experience building {phrase} systems."
    if career_evidence >= 0.35:
        return f"Career history includes work on {phrase} systems."
    return f"Some exposure to {phrase} in past roles."


def _behavioral_sentence(candidate: dict[str, Any], features) -> str:
    signals = candidate.get("redrob_signals", {})
    response_rate = float(signals.get("recruiter_response_rate") or 0.0) * 100
    open_to_work = bool(signals.get("open_to_work_flag"))
    last_active = signals.get("last_active_date")

    if response_rate >= 70 and open_to_work:
        return "Actively available with a strong recruiter response rate."
    if response_rate >= 70:
        return f"Strong recruiter response rate ({response_rate:.0f}%)."
    if open_to_work and last_active:
        return f"Active on the platform as of {last_active} and open to work."
    if open_to_work:
        return "Flagged open to work."
    if response_rate >= 40:
        return f"Recruiter response rate is {response_rate:.0f}%."
    return f"Low recruiter response rate ({response_rate:.0f}%) — may be harder to reach."


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
        _evidence_sentence(features),
        _behavioral_sentence(candidate, features),
    ]
    concern = _concern_sentence(candidate, features)
    if concern:
        parts.append(concern)
    return " ".join(parts)
