# how good is the cndidate for the job?
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import math
import re


PRODUCT_COMPANY_HINTS = {
    "google", "meta", "amazon", "microsoft", "uber", "airbnb", "flipkart", "linkedin",
    "spotify", "netflix", "zomato", "swiggy", "adobe", "atlassian", "salesforce"
}

TECH_EVIDENCE_TERMS = {
    "ranking", "ranker", "retrieval", "search", "recommendation", "recommender",
    "embedding", "embeddings", "vector", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "elasticsearch", "opensearch", "bm25", "llm", "fine-tuning", "fine tuning",
    "production", "deployed", "evaluation", "ndcg", "mrr", "map", "offline"
}

BEHAVIOR_POSITIVE = {
    "open_to_work_flag",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
}


@dataclass
class CandidateFeatures:
    candidate_id: str
    years_of_experience: float
    technical_domain_fit: float
    career_evidence_fit: float
    production_experience_fit: float
    behavioral_fit: float
    experience_alignment: float
    logistics_fit: float
    honeypot_risk: float
    final_score: float
    matched_evidence: list[str]
    concerns: list[str]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _count_term_hits(text: str, terms: set[str] | list[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _date_to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _total_career_months(candidate: dict[str, Any]) -> int:
    total = 0
    for role in candidate.get("career_history", []):
        total += int(role.get("duration_months") or 0)
    return total


def _job_title_fit(candidate: dict[str, Any]) -> float:
    title = _normalize_text(candidate["profile"].get("current_title", ""))
    seniority_markers = ["engineer", "scientist", "ml", "ai", "search", "data", "backend", "platform"]
    hits = sum(1 for marker in seniority_markers if marker in title)
    return _clamp(hits / len(seniority_markers))


def _career_evidence_score(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    texts: list[str] = []
    for role in candidate.get("career_history", []):
        texts.append(role.get("title", ""))
        texts.append(role.get("description", ""))
    profile = candidate.get("profile", {})
    texts.append(profile.get("headline", ""))
    texts.append(profile.get("summary", ""))
    joined = " ".join(texts).lower()

    evidence_terms = [
        "ranking",
        "ranker",
        "retrieval",
        "search",
        "recommendation",
        "recommender",
        "embedding",
        "embeddings",
        "vector",
        "faiss",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "elasticsearch",
        "opensearch",
        "evaluation",
        "production",
        "deployed",
    ]
    hits = [term for term in evidence_terms if term in joined]

    project_verbs = ["built", "designed", "implemented", "shipped", "owned", "led", "deployed", "improved"]
    project_hits = [verb for verb in project_verbs if verb in joined]

    score = 0.0
    score += min(len(hits), 6) / 6.0 * 0.7
    score += min(len(project_hits), 5) / 5.0 * 0.3

    evidence_snippets: list[str] = []
    for role in candidate.get("career_history", []):
        role_text = f"{role.get('title', '')} {role.get('description', '')}".lower()
        if any(term in role_text for term in evidence_terms):
            evidence_snippets.append(f"{role.get('title', '')} at {role.get('company', '')}")
    if not evidence_snippets:
        for skill in candidate.get("skills", [])[:5]:
            name = skill.get("name", "")
            if name:
                evidence_snippets.append(name)
    return _clamp(score), evidence_snippets[:3]


def _technical_domain_score(candidate: dict[str, Any], jd_terms: list[str]) -> float:
    texts = []
    profile = candidate.get("profile", {})
    texts.extend([profile.get("headline", ""), profile.get("summary", "")])
    for role in candidate.get("career_history", []):
        texts.extend([role.get("title", ""), role.get("description", "")])
    for skill in candidate.get("skills", []):
        texts.append(skill.get("name", ""))
    joined = " ".join(texts).lower()
    hits = sum(1 for term in jd_terms if term in joined)
    return _clamp(hits / max(5, len(jd_terms) or 1))


def _production_experience_score(candidate: dict[str, Any]) -> float:
    texts = []
    for role in candidate.get("career_history", []):
        texts.append(role.get("description", ""))
    joined = " ".join(texts).lower()
    signals = [
        "production",
        "deployed",
        "real-time",
        "pipeline",
        "scale",
        "users",
        "applied",
        "owned",
        "operational",
        "ab testing",
        "a/b",
    ]
    hits = sum(1 for term in signals if term in joined)
    return _clamp(hits / len(signals))


def _behavioral_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    completeness = float(signals.get("profile_completeness_score", 0.0)) / 100.0
    response_rate = float(signals.get("recruiter_response_rate", 0.0))
    interview_completion = float(signals.get("interview_completion_rate", 0.0))
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    recent_activity = 1.0 if signals.get("last_active_date") else 0.0
    verified = sum(1 for key in BEHAVIOR_POSITIVE if signals.get(key)) / len(BEHAVIOR_POSITIVE)
    github = float(signals.get("github_activity_score", -1))
    github_score = 0.0 if github < 0 else github / 100.0

    score += completeness * 0.20
    score += response_rate * 0.25
    score += interview_completion * 0.20
    score += open_to_work * 0.20
    score += recent_activity * 0.05
    score += verified * 0.05
    score += github_score * 0.05

    return _clamp(score)


def _experience_alignment_score(candidate: dict[str, Any], jd_years_min: float, jd_years_max: float) -> float:
    years = float(candidate["profile"].get("years_of_experience", 0.0))
    if jd_years_min <= 0 and jd_years_max <= 0:
        return 0.5
    if years < jd_years_min:
        return _clamp(years / max(1.0, jd_years_min))
    if years > jd_years_max:
        over = years - jd_years_max
        return _clamp(1.0 - min(over / max(1.0, jd_years_max), 0.5))
    midpoint = (jd_years_min + jd_years_max) / 2.0
    return _clamp(1.0 - abs(years - midpoint) / max(1.0, (jd_years_max - jd_years_min) / 2.0 + 1.0))


def _logistics_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    score = 0.0
    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in {"hybrid", "flexible"}:
        score += 0.4
    if signals.get("willing_to_relocate"):
        score += 0.3
    notice = int(signals.get("notice_period_days", 180))
    if notice <= 30:
        score += 0.3
    elif notice <= 60:
        score += 0.2
    elif notice <= 90:
        score += 0.1
    return _clamp(score)


def _honeypot_risk(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    profile = candidate.get("profile", {})
    years = float(profile.get("years_of_experience", 0.0))
    career_months = _total_career_months(candidate)
    skill_months = [
        int(skill.get("duration_months") or 0)
        for skill in candidate.get("skills", [])
        if skill.get("duration_months") is not None
    ]
    max_skill = max(skill_months) if skill_months else 0
    title = _normalize_text(profile.get("current_title", ""))
    signals = candidate.get("redrob_signals", {})

    risk = 0.0
    concerns: list[str] = []

    if career_months > years * 12 + 24:
        risk += 0.35
        concerns.append("career history looks longer than claimed experience")
    if years < 3 and any(marker in title for marker in ["principal", "staff", "architect", "head", "lead"]):
        risk += 0.30
        concerns.append("seniority in title looks too high for experience")
    if max_skill > years * 12 + 12 and max_skill > 0:
        risk += 0.25
        concerns.append("skill duration looks too high for total experience")
    if not signals.get("open_to_work_flag") and int(signals.get("applications_submitted_30d", 0)) > 20:
        risk += 0.15
        concerns.append("activity and open-to-work status look inconsistent")
    if float(signals.get("recruiter_response_rate", 0.0)) < 0.05 and int(signals.get("saved_by_recruiters_30d", 0)) > 5:
        risk += 0.10
        concerns.append("strong recruiter interest but very low response rate")

    return _clamp(risk), concerns

# main function 
def score_candidate(candidate: dict[str, Any], jd_years_min: float, jd_years_max: float, jd_terms: list[str]) -> CandidateFeatures:
    years = float(candidate["profile"].get("years_of_experience", 0.0))
    technical = _technical_domain_score(candidate, jd_terms)
    career_evidence, evidence_snippets = _career_evidence_score(candidate)
    production = _production_experience_score(candidate)
    behavioral = _behavioral_score(candidate)
    experience_alignment = _experience_alignment_score(candidate, jd_years_min, jd_years_max)
    logistics = _logistics_score(candidate)
    honeypot_risk, concerns = _honeypot_risk(candidate)

    final_score = (
        0.25 * technical
        + 0.20 * career_evidence
        + 0.20 * production
        + 0.15 * behavioral
        + 0.10 * experience_alignment
        + 0.10 * logistics
    )
    final_score = final_score * (1.0 - 0.65 * honeypot_risk) # applying honeypot risk penalty

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        years_of_experience=years,
        technical_domain_fit=technical,
        career_evidence_fit=career_evidence,
        production_experience_fit=production,
        behavioral_fit=behavioral,
        experience_alignment=experience_alignment,
        logistics_fit=logistics,
        honeypot_risk=honeypot_risk,
        final_score=_clamp(final_score),
        matched_evidence=evidence_snippets,
        concerns=concerns,
    )

