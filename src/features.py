from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import re

from .evidence import extract_evidence
from .tfidf import IDFModel, concept_expand, cosine_similarity, tfidf_vector, tokenize


PRODUCT_COMPANY_HINTS = {
    "google", "meta", "amazon", "microsoft", "uber", "airbnb", "flipkart", "linkedin",
    "spotify", "netflix", "zomato", "swiggy", "adobe", "atlassian", "salesforce"
}

CONSULTING_FIRMS = {
    "tata consultancy", "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl tech",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "mindtree",
}

INDIA_PREFERRED_CITIES = {"pune", "noida"}
INDIA_ACCEPTABLE_CITIES = {
    "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
    "bengaluru", "bangalore", "chennai",
}

# 180 days is the JD's own "6-month" bar for activity decay.
ACTIVITY_INACTIVE_THRESHOLD_DAYS = 180

# p95 instead of max: a few extreme outliers otherwise compress everyone else near 0.
DEMAND_PERCENTILE = 0.95


@dataclass
class DemandCaps:
    saved_cap: float
    views_cap: float
    search_cap: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(pct * len(ordered)))
    return ordered[index]


def extract_demand_raw(candidate: dict[str, Any]) -> tuple[float, float, float]:
    signals = candidate.get("redrob_signals", {})
    return (
        float(signals.get("saved_by_recruiters_30d") or 0),
        float(signals.get("profile_views_received_30d") or 0),
        float(signals.get("search_appearance_30d") or 0),
    )


def learn_demand_caps(raw_rows: Any) -> DemandCaps:
    saved_vals: list[float] = []
    views_vals: list[float] = []
    search_vals: list[float] = []
    for saved, views, search in raw_rows:
        saved_vals.append(saved)
        views_vals.append(views)
        search_vals.append(search)
    return DemandCaps(
        saved_cap=_percentile(saved_vals, DEMAND_PERCENTILE),
        views_cap=_percentile(views_vals, DEMAND_PERCENTILE),
        search_cap=_percentile(search_vals, DEMAND_PERCENTILE),
    )


@dataclass
class CandidateFeatures:
    candidate_id: str
    years_of_experience: float
    capability_match: float
    career_evidence_fit: float
    production_experience_fit: float
    behavioral_fit: float
    experience_alignment: float
    logistics_fit: float
    recruiter_demand: float
    honeypot_risk: float
    disqualifier_risk: float
    final_score: float
    matched_evidence: list[str]
    concerns: list[str]
    disqualifier_concerns: list[str]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _date_to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def extract_last_active_date(candidate: dict[str, Any]) -> date | None:
    # Anchored to pool's own max date, not wall-clock, so output stays deterministic.
    signals = candidate.get("redrob_signals", {})
    return _date_to_date(signals.get("last_active_date"))


def _total_career_months(candidate: dict[str, Any]) -> int:
    total = 0
    for role in candidate.get("career_history", []):
        total += int(role.get("duration_months") or 0)
    return total


def _phrase_idf_weight(phrase: str, idf: IDFModel) -> float:
    tokens = tokenize(phrase)
    if not tokens:
        return 0.0
    return sum(idf.weight(tok) for tok in tokens) / len(tokens)


def _is_consulting_company(company: str) -> bool:
    lower = company.lower()
    return any(firm in lower for firm in CONSULTING_FIRMS)


def _has_product_company_experience(candidate: dict[str, Any]) -> bool:
    for role in candidate.get("career_history", []):
        company = role.get("company", "").lower()
        if any(hint in company for hint in PRODUCT_COMPANY_HINTS):
            return True
    return False


def _location_score(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    combined = f"{profile.get('location') or ''} {profile.get('country') or ''}".lower()
    if any(city in combined for city in INDIA_PREFERRED_CITIES):
        return 1.0
    if any(city in combined for city in INDIA_ACCEPTABLE_CITIES):
        return 0.75
    if "india" in combined:
        return 0.50
    return 0.0


def _response_time_score(hours: float | None) -> float:
    if hours is None or hours < 0:
        return 0.5
    if hours <= 24:
        return 1.0
    if hours <= 72:
        return 0.7
    if hours <= 168:
        return 0.4
    return 0.1


def _skill_assessment_bonus(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores") or {}
    relevant: list[float] = []
    for skill, score in assessments.items():
        if any(term in skill.lower() for term in {
            "nlp", "retrieval", "ranking", "search", "recommendation",
            "embedding", "machine learning", "llm", "fine-tuning",
            "vector", "information retrieval",
        }):
            relevant.append(float(score) / 100.0)
    return sum(relevant) / len(relevant) if relevant else 0.0


def _career_evidence_score(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    result = extract_evidence(candidate)
    score = result.score
    if _has_product_company_experience(candidate):
        score = min(1.0, score + 0.10)
    return _clamp(score), result.domains_found


def candidate_profile_text(candidate: dict[str, Any]) -> str:
    texts = []
    profile = candidate.get("profile", {})
    texts.extend([profile.get("headline", ""), profile.get("summary", "")])
    for role in candidate.get("career_history", []):
        texts.extend([role.get("title", ""), role.get("description", "")])
    for skill in candidate.get("skills", []):
        texts.append(skill.get("name", ""))
    return " ".join(texts)


def _capability_match_score(candidate: dict[str, Any], idf: IDFModel, jd_vector: dict[str, float]) -> float:
    tokens = tokenize(concept_expand(candidate_profile_text(candidate)))
    candidate_vector = tfidf_vector(tokens, idf)
    tfidf_sim = _clamp(cosine_similarity(jd_vector, candidate_vector))
    assessment = _skill_assessment_bonus(candidate)
    return _clamp(0.85 * tfidf_sim + 0.15 * assessment)


def _production_experience_score(candidate: dict[str, Any], idf: IDFModel) -> float:
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
    matched = [term for term in signals if term in joined]
    total_weight = sum(_phrase_idf_weight(term, idf) for term in signals)
    matched_weight = sum(_phrase_idf_weight(term, idf) for term in matched)
    return _clamp(matched_weight / total_weight) if total_weight > 0 else 0.0


def _activity_score(candidate: dict[str, Any], dataset_as_of: date) -> float:
    signals = candidate.get("redrob_signals", {})
    last_active = extract_last_active_date(candidate)
    if last_active is None:
        recency_score = 0.0
    else:
        days_inactive = max(0, (dataset_as_of - last_active).days)
        recency_score = _clamp(1.0 - days_inactive / ACTIVITY_INACTIVE_THRESHOLD_DAYS)

    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    return _clamp(0.7 * recency_score + 0.3 * open_to_work)


def _behavioral_score(candidate: dict[str, Any], dataset_as_of: date) -> float:
    signals = candidate.get("redrob_signals", {})

    activity_score = _activity_score(candidate, dataset_as_of)
    response_score = _clamp(float(signals.get("recruiter_response_rate", 0.0)))
    reliability_score = _clamp(float(signals.get("interview_completion_rate", 0.0)))

    rt_raw = signals.get("avg_response_time_hours")
    response_time_score = _response_time_score(float(rt_raw) if rt_raw is not None else None)

    offer_raw = float(signals.get("offer_acceptance_rate", 0.0))
    offer_acceptance = _clamp(offer_raw) if offer_raw >= 0 else 0.5  # -1 means no prior offers → neutral

    return _clamp(
        0.30 * activity_score
        + 0.25 * response_score
        + 0.20 * reliability_score
        + 0.15 * response_time_score
        + 0.10 * offer_acceptance
    )


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

    score += _location_score(candidate) * 0.30

    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in {"hybrid", "flexible"}:
        score += 0.25

    if signals.get("willing_to_relocate"):
        score += 0.20

    notice = int(signals.get("notice_period_days", 180))
    if notice <= 30:
        score += 0.25
    elif notice <= 60:
        score += 0.15
    elif notice <= 90:
        score += 0.05

    return _clamp(score)


def _recruiter_demand_score(candidate: dict[str, Any], demand_caps: DemandCaps) -> float:
    saved, views, search = extract_demand_raw(candidate)

    def _ratio(value: float, cap: float) -> float:
        return _clamp(value / cap) if cap > 0 else 0.0

    saved_score = _ratio(saved, demand_caps.saved_cap)
    views_score = _ratio(views, demand_caps.views_cap)
    search_score = _ratio(search, demand_caps.search_cap)
    return _clamp((saved_score + views_score + search_score) / 3.0)


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

    # Expert proficiency with ~0 months of use is the classic honeypot signature
    # (confirmed: ~20 candidates in this dataset have 3-5 such skills, everyone else has 0).
    expert_zero_duration = sum(
        1
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert" and (skill.get("duration_months") or 0) <= 2
    )
    if expert_zero_duration >= 2:
        risk += 0.35
        concerns.append(
            f"claims expert-level proficiency on {expert_zero_duration} skills with ~0 months of use"
        )

    return _clamp(risk), concerns


def _disqualifier_risk(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    risk = 0.0
    concerns: list[str] = []
    roles = candidate.get("career_history", [])

    if roles:
        total_months = sum(int(r.get("duration_months") or 0) for r in roles)
        if total_months >= 24 and all(_is_consulting_company(r.get("company", "")) for r in roles):
            risk += 0.20  # preference, not hard rejection
            concerns.append("entire career history is at IT services / consulting firms")

    FRAMEWORK_TOOLS = {
        "langchain", "llamaindex", "llama-index", "llama index",
        "autogpt", "auto-gpt", "crewai", "crew ai", "flowise",
        "langflow", "agentgpt",
    }
    RETRIEVAL_DEPTH_TERMS = {
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
        "opensearch", "bm25", "retrieval", "ranking", "recommendation",
        "fine-tuning", "fine tuning", "ndcg", "mrr",
    }
    profile_text = candidate_profile_text(candidate).lower()
    matched_frameworks = [f for f in FRAMEWORK_TOOLS if f in profile_text]
    if matched_frameworks:
        has_retrieval_depth = any(term in profile_text for term in RETRIEVAL_DEPTH_TERMS)
        if not has_retrieval_depth:
            risk += 0.25
            framework_str = ", ".join(matched_frameworks[:2])
            concerns.append(
                f"AI experience appears limited to LLM orchestration frameworks "
                f"({framework_str}) without retrieval or ranking background"
            )

    if roles:
        short_stints = sum(1 for r in roles if int(r.get("duration_months") or 0) < 18)
        if len(roles) >= 3 and short_stints >= 3:
            risk += 0.20
            concerns.append(
                f"{short_stints} of {len(roles)} roles under 18 months — "
                "possible title-chasing pattern"
            )

    CV_SPEECH_ROBOTICS_TERMS = {
        "computer vision", "image classification", "object detection",
        "image segmentation", "convolutional neural", "opencv",
        "speech recognition", "asr", "text-to-speech", "tts",
        "speech synthesis", "wav2vec", "whisper",
        "robotics", "lidar", "autonomous driving", "slam", "point cloud",
    }
    NLP_IR_TERMS = {
        "nlp", "natural language", "information retrieval", "retrieval",
        "ranking", "recommendation", "search", "semantic search",
        "question answering", "text classification", "named entity",
        "ndcg", "mrr", "bm25",
    }
    matched_cv = sum(1 for t in CV_SPEECH_ROBOTICS_TERMS if t in profile_text)
    has_nlp_ir = any(t in profile_text for t in NLP_IR_TERMS)
    if matched_cv >= 2 and not has_nlp_ir:
        risk += 0.30
        concerns.append(
            "primary expertise appears to be computer vision, speech, or robotics "
            "without NLP or information retrieval exposure"
        )

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    if "india" not in country and "india" not in location:
        if not signals.get("willing_to_relocate"):
            risk += 0.35
            concerns.append(
                f"based in {profile.get('location', 'outside India')} "
                "and not open to relocation — no visa sponsorship available"
            )

    return _clamp(risk), concerns


def score_candidate(
    candidate: dict[str, Any],
    jd_years_min: float,
    jd_years_max: float,
    idf: IDFModel,
    jd_vector: dict[str, float],
    dataset_as_of: date,
    demand_caps: DemandCaps,
) -> CandidateFeatures:
    years = float(candidate["profile"].get("years_of_experience", 0.0))
    capability_match = _capability_match_score(candidate, idf, jd_vector)
    career_evidence, evidence_snippets = _career_evidence_score(candidate)
    production = _production_experience_score(candidate, idf)
    behavioral = _behavioral_score(candidate, dataset_as_of)
    experience_alignment = _experience_alignment_score(candidate, jd_years_min, jd_years_max)
    logistics = _logistics_score(candidate)
    recruiter_demand = _recruiter_demand_score(candidate, demand_caps)
    honeypot_risk, concerns = _honeypot_risk(candidate)
    disqualifier_risk, disqualifier_concerns = _disqualifier_risk(candidate)

    final_score = (
        0.10 * capability_match
        + 0.25 * career_evidence
        + 0.19 * production
        + 0.18 * behavioral
        + 0.09 * experience_alignment
        + 0.09 * logistics
        + 0.10 * recruiter_demand
    )
    final_score = final_score * (1.0 - 0.65 * honeypot_risk) * (1.0 - 0.40 * disqualifier_risk)

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        years_of_experience=years,
        capability_match=capability_match,
        career_evidence_fit=career_evidence,
        production_experience_fit=production,
        behavioral_fit=behavioral,
        experience_alignment=experience_alignment,
        logistics_fit=logistics,
        recruiter_demand=recruiter_demand,
        honeypot_risk=honeypot_risk,
        disqualifier_risk=disqualifier_risk,
        final_score=_clamp(final_score),
        matched_evidence=evidence_snippets,
        concerns=concerns,
        disqualifier_concerns=disqualifier_concerns,
    )
