from __future__ import annotations

from typing import Any


def _variant(candidate_id: str, modulus: int) -> int:
    """Deterministic per-candidate variant so phrasing differs across candidates
    without breaking reproducibility (no randomness, no hash())."""
    digits = "".join(ch for ch in candidate_id if ch.isdigit())
    return (int(digits) if digits else 0) % modulus


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
    """Pick the most specific domain phrase that matches the candidate's profile."""
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
    """Sentence 1: role + years + domain."""
    profile = candidate.get("profile", {})
    title = profile.get("current_title") or "Engineer"
    years = features.years_of_experience
    domain = _domain_phrase(candidate)
    return f"{title} with {years} years of experience building {domain}."


def _format_evidence_snippet(snippet: str) -> tuple[str, bool]:
    """Parse a matched_evidence entry.

    Returns (readable_clause, is_verb_phrase) where is_verb_phrase is True
    for builder-phrase entries (e.g. '"Built ranking pipeline" at Company')
    and False for role titles or skill lists.
    """
    if snippet.startswith('"'):
        # Builder phrase: '"Verb phrase text" at Company'
        if '" at ' in snippet:
            phrase, company = snippet.split('" at ', 1)
            # Only lowercase the verb clause; preserve company name casing.
            return phrase.lstrip('"').lower() + " at " + company.strip(), True
        return snippet.strip('"').lower(), True
    return snippet, False


def _evidence_and_behavioral_sentence(candidate: dict[str, Any], features) -> str:
    """Sentence 2: career evidence + behavioral/availability signal in one sentence."""
    evidence = features.matched_evidence
    signals = candidate.get("redrob_signals", {})

    response_rate = float(signals.get("recruiter_response_rate") or 0.0) * 100
    open_to_work = bool(signals.get("open_to_work_flag"))
    last_active = signals.get("last_active_date")

    # --- Evidence clause ---
    v = _variant(candidate.get("candidate_id", ""), 3)

    if evidence:
        first, is_verb = _format_evidence_snippet(evidence[0])
        second_clause = ""
        if len(evidence) > 1:
            second, _ = _format_evidence_snippet(evidence[1])
            second_clause = f" and {second}"

        if is_verb:
            # Verb-phrase evidence: "Has built X at Company and shipped Y at Company"
            prefixes = ["Has ", "Directly ", "Has directly "]
            ev_part = prefixes[v] + first + second_clause
        else:
            # Role-title or skill evidence: "Demonstrates X through role/skill"
            prefixes = [
                "Demonstrates production-scale ownership through ",
                "Shows hands-on depth through ",
                "Demonstrates relevant experience through ",
            ]
            ev_part = prefixes[v] + first + second_clause
    else:
        profile = candidate.get("profile", {})
        company = profile.get("current_company") or "their current employer"
        ev_part = f"Background includes relevant technical work at {company}"

    # --- Behavioral clause ---
    if response_rate >= 70 and open_to_work:
        beh = "remains highly engaged on the platform with a strong recruiter response rate"
    elif response_rate >= 70:
        beh = f"maintains a strong recruiter response rate ({response_rate:.0f}%)"
    elif open_to_work and last_active:
        beh = f"active on the platform as of {last_active} and flagged open to work"
    elif open_to_work:
        beh = "flagged open to work"
    elif response_rate >= 40:
        beh = f"recruiter response rate of {response_rate:.0f}%"
    else:
        beh = f"low recruiter response rate ({response_rate:.0f}%)"

    return f"{ev_part}, and {beh}."


def _concern_sentence(candidate: dict[str, Any], features) -> str | None:
    """Sentence 3 (optional): one honest concern, in priority order."""
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # JD fit disqualifiers (consulting-only, LangChain-only, title-chaser)
    if features.disqualifier_concerns:
        c = features.disqualifier_concerns[0]
        return (c[0].upper() + c[1:]).rstrip(".") + "."

    # Honeypot / profile inconsistency
    if features.concerns:
        c = features.concerns[0]
        return (c[0].upper() + c[1:]).rstrip(".") + "."

    # Notice period > 60 days
    notice = signals.get("notice_period_days")
    if notice is not None and int(notice) > 60:
        return f"Notice period of {notice} days may slow hiring."

    # Based outside India and not open to relocate
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
