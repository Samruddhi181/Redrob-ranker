# how good is the cndidate for the job?
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import math
import re

from .tfidf import IDFModel, cosine_similarity, tfidf_vector, tokenize


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

# Action verbs that, on their own, mean almost nothing (everyone's resume
# says "built" or "led" something) -- but paired with a JD-relevant noun a
# few words later, become real proof of having actually done the work
# rather than just having it on a skill list. See _find_builder_phrases.
BUILDER_VERBS = {
    "built", "build", "designed", "design", "developed", "develop",
    "implemented", "implement", "shipped", "ship", "launched", "launch",
    "architected", "architect", "engineered", "engineer", "led", "lead",
    "drove", "drive", "created", "create", "owned", "own",
    "improved", "improve", "optimized", "optimize", "scaled", "scale",
    "delivered", "deliver",
}

BUILDER_PHRASE_WINDOW = 6  # words scanned after the verb, looking for a domain term
BUILDER_PHRASE_TRAILING_WORDS = 2  # extra words kept so the phrase reads complete, e.g. "...recommendation engine" not just "...recommendation"
BUILDER_PHRASE_TRAILING_STOPWORDS = {
    "for", "from", "the", "a", "an", "to", "and", "or", "of", "in", "on", "at", "by", "with",
}  # trimmed off the end so a phrase doesn't dangle on a stray preposition/article

# Graded, not flat: the first concrete match already earns most of the
# bonus (that's the "huge bonus" -- one real "built X ranking system" line
# is worth far more than any number of loose keyword hits), with smaller
# returns for additional matches so restating the same achievement a few
# different ways can't be gamed into a multiplied score. 3+ matches -> 1.0.
BUILDER_PHRASE_BONUS_BY_COUNT = {0: 0.0, 1: 0.6, 2: 0.85}

# IT services / consulting firms the JD explicitly names as a disqualifier
# when they make up a candidate's *entire* career history.
CONSULTING_FIRMS = {
    "tata consultancy", "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl tech",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "mindtree",
}

# JD location preferences: Pune/Noida ideal, other tier-1 Indian cities acceptable.
INDIA_PREFERRED_CITIES = {"pune", "noida"}
INDIA_ACCEPTABLE_CITIES = {
    "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
    "bengaluru", "bangalore", "chennai",
}

# The JD's own words (closing note to participants): "a perfect-on-paper
# candidate who hasn't logged in for 6 months and has a 5% recruiter
# response rate is, for hiring purposes, not actually available. Down-weight
# them appropriately." 180 days is that 6-month bar, used as the point past
# which activity_score bottoms out at 0 rather than decaying forever.
ACTIVITY_INACTIVE_THRESHOLD_DAYS = 180

# Recruiter-demand fields, all unbounded right-skewed 30-day counts: "how
# much do recruiters already like this candidate", independent of anything
# the candidate themselves wrote or did. A few outliers run far above the
# typical candidate (e.g. search_appearance_30d: p95=278 but max=1490 on
# this dataset), so a raw-max normalization would compress everyone else
# into a tiny sliver near 0. Instead each field is normalized against its
# own corpus-wide 95th percentile, learned in the same streaming first pass
# that already learns IDF weights and the activity "as of" date (see
# DemandCaps/learn_demand_caps below and _profile_tokens_tracking_activity
# in ranker.py) -- the top ~5% of candidates on a given signal all earn full
# credit on it, and everyone else is scored relative to that real ceiling
# instead of one or two extreme profiles.
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
    """Public so ranker.py can stream over the whole pool once (the same
    pass that already learns IDF weights and the activity as-of date) and
    collect the raw values needed to learn each field's corpus-wide p95 cap
    -- see learn_demand_caps."""
    signals = candidate.get("redrob_signals", {})
    return (
        float(signals.get("saved_by_recruiters_30d") or 0),
        float(signals.get("profile_views_received_30d") or 0),
        float(signals.get("search_appearance_30d") or 0),
    )


def learn_demand_caps(raw_rows: Any) -> DemandCaps:
    """raw_rows is an iterable of (saved, views, search) tuples, one per
    candidate, gathered via extract_demand_raw during ranker.py's first
    pass. Returns the corpus's own 95th-percentile value for each field,
    used as the normalization ceiling in _recruiter_demand_score -- "learn
    it from the corpus itself" rather than picking an arbitrary scale."""
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


def extract_last_active_date(candidate: dict[str, Any]) -> date | None:
    """Public so ranker.py can stream over the whole pool once and find the
    most recent last_active_date across all 100K candidates -- used as the
    "as of" anchor for everyone's activity_score (see ranker.py), instead of
    wall-clock time. The dataset doesn't carry its own as-of date, and using
    date.today() would make scores (and the submitted CSV) drift depending
    on what day the script happens to run, breaking the byte-identical
    reproducibility this pipeline otherwise guarantees. Anchoring to the
    pool's own freshest signal keeps it fully deterministic and self
    contained, the same way IDF weights are learned from the corpus itself
    rather than an external source."""
    signals = candidate.get("redrob_signals", {})
    return _date_to_date(signals.get("last_active_date"))


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


def _phrase_idf_weight(phrase: str, idf: IDFModel) -> float:
    """Average corpus IDF weight of a (possibly multi-word) signal phrase.
    Used to make matching a rare, distinctive phrase (e.g. "qdrant", "a/b")
    count for more than matching a phrase nearly everyone's profile uses
    (e.g. "production") -- an exact match is still required, but not every
    exact match is treated as equally meaningful anymore."""
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
    """Converts avg_response_time_hours to a 0-1 score. Lower is better."""
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
    """Average platform-verified assessment score (0-1) across JD-relevant skills.
    These are scored by Redrob's own platform, not self-reported, so they add
    signal that's orthogonal to what the candidate wrote in their profile text."""
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


def _find_builder_phrases(text: str) -> list[str]:
    """Scan one text chunk (a single role description, headline, or summary
    -- never the whole profile joined together, so a verb from one role
    can't pair with a noun from a different one) for a builder verb
    followed within a few words by one of the JD-relevant domain terms --
    e.g. "Built candidate recommendation engine" or "Led search relevance
    improvements". A verb-plus-nearby-domain-noun match like this is a much
    rarer, much more specific signal than either half alone (a builder verb
    anywhere, or a domain term anywhere in the profile), which is why it
    earns a steep bonus in _career_evidence_score instead of counting the
    same as a loose keyword hit.
    """
    if not text:
        return []
    words = re.findall(r"[A-Za-z][A-Za-z0-9/+\-']*", text)
    lowered = [w.lower() for w in words]
    phrases: list[str] = []
    for i, word in enumerate(lowered):
        if word not in BUILDER_VERBS:
            continue
        window_end = min(len(words), i + 1 + BUILDER_PHRASE_WINDOW)
        match_end = None
        for k in range(i + 1, window_end):
            span = " ".join(lowered[i + 1 : k + 1])
            if any(term in span for term in TECH_EVIDENCE_TERMS):
                match_end = k + 1
                break
        if match_end is None:
            continue
        display_end = min(len(words), match_end + BUILDER_PHRASE_TRAILING_WORDS)
        while display_end > match_end and lowered[display_end - 1] in BUILDER_PHRASE_TRAILING_STOPWORDS:
            display_end -= 1
        phrases.append(" ".join(words[i:display_end]))
    return phrases


def _builder_phrase_evidence(candidate: dict[str, Any]) -> list[tuple[str, str]]:
    """Distinct (phrase, company) builder-phrase matches across
    career_history[].description, profile.summary, and profile.headline --
    deliberately not role titles, which name a job, not a claim of what was
    built. Company is "" for summary/headline matches, which have none."""
    profile = candidate.get("profile", {})
    found: list[tuple[str, str]] = []
    for role in candidate.get("career_history", []):
        company = role.get("company", "")
        for phrase in _find_builder_phrases(role.get("description", "")):
            found.append((phrase, company))
    for phrase in _find_builder_phrases(profile.get("summary", "")):
        found.append((phrase, ""))
    for phrase in _find_builder_phrases(profile.get("headline", "")):
        found.append((phrase, ""))

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for phrase, company in found:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((phrase, company))
    return deduped


def _career_evidence_score(candidate: dict[str, Any], idf: IDFModel) -> tuple[float, list[str]]:
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

    # IDF-weighted hit ratio rather than a flat hit count capped at 6: a
    # candidate whose evidence is "qdrant" and "ranker" (rare, JD-specific)
    # now outscores one whose only hits are "production" and "deployed"
    # (common almost everywhere), instead of the two counting identically.
    total_weight = sum(_phrase_idf_weight(term, idf) for term in evidence_terms)
    matched_weight = sum(_phrase_idf_weight(term, idf) for term in hits)
    evidence_component = matched_weight / total_weight if total_weight > 0 else 0.0

    # The biggest single signal: did the candidate's own career history,
    # summary, or headline actually name a concrete thing they built that's
    # relevant to this JD -- "Built candidate recommendation engine", "Led
    # search relevance improvements" -- rather than just listing a skill or
    # using a builder verb somewhere unrelated. See _find_builder_phrases.
    builder_matches = _builder_phrase_evidence(candidate)
    builder_bonus = BUILDER_PHRASE_BONUS_BY_COUNT.get(len(builder_matches), 1.0)

    score = 0.0
    score += builder_bonus * 0.55
    score += _clamp(evidence_component) * 0.30
    score += min(len(project_hits), 5) / 5.0 * 0.15

    # Product-company experience bonus: the JD explicitly values candidates who
    # have shipped at product companies over consulting-only backgrounds. One
    # known product company anywhere in career history earns a 0.10 boost,
    # capped at 1.0 so it can't inflate a genuinely weak evidence score.
    if _has_product_company_experience(candidate):
        score = min(1.0, score + 0.10)

    evidence_snippets: list[str] = []
    for phrase, company in builder_matches[:3]:
        if company:
            evidence_snippets.append(f'"{phrase}" at {company}')
        else:
            evidence_snippets.append(f'"{phrase}"')
    if not evidence_snippets:
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


def candidate_profile_text(candidate: dict[str, Any]) -> str:
    """Text surface used for JD matching: headline, summary, role titles and
    descriptions, skill names. Shared by the corpus-wide IDF pass in
    ranker.py and the per-candidate score below, so both see exactly the
    same text."""
    texts = []
    profile = candidate.get("profile", {})
    texts.extend([profile.get("headline", ""), profile.get("summary", "")])
    for role in candidate.get("career_history", []):
        texts.extend([role.get("title", ""), role.get("description", "")])
    for skill in candidate.get("skills", []):
        texts.append(skill.get("name", ""))
    return " ".join(texts)


def _capability_match_score(candidate: dict[str, Any], idf: IDFModel, jd_vector: dict[str, float]) -> float:
    """TF-IDF cosine similarity between the candidate's profile and the JD,
    blended with Redrob's own platform-verified skill assessments on
    JD-relevant skills. The TF-IDF component captures vocabulary overlap
    (rarer, more JD-specific words count more); the assessment component adds
    independently-verified signal that isn't limited to what the candidate
    chose to write -- a candidate who scored 85/100 on an NLP assessment is
    more credible than one who only lists "NLP" in their skills section."""
    tokens = tokenize(candidate_profile_text(candidate))
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

    # Same IDF-weighting idea as career evidence: a hit on a signal almost
    # every description uses ("users", "production") counts for less than a
    # hit on one that's actually rare in this candidate pool ("real-time",
    # "a/b"), instead of every exact match being worth the same one point.
    total_weight = sum(_phrase_idf_weight(term, idf) for term in signals)
    matched_weight = sum(_phrase_idf_weight(term, idf) for term in matched)
    return _clamp(matched_weight / total_weight) if total_weight > 0 else 0.0


def _activity_score(candidate: dict[str, Any], dataset_as_of: date) -> float:
    """Is this candidate actually reachable right now? The JD's own example
    of a "not actually available" candidate combines two things: hasn't
    logged in for 6 months, and not flagged open to work ("Active on Redrob
    platform (or has clear signal of being in the job market) so we can
    actually talk to them"). Recency decays linearly from full credit at 0
    days idle to 0 credit at the JD's own 180-day (6-month) bar, instead of
    the old binary "does this field exist" check, which gave a candidate
    idle 5 months the exact same credit as one active yesterday."""
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
    """Redrob engagement across five signals, weighted to keep the JD's
    explicitly-called-out dimensions (activity, response rate) at the front.
    activity + recruiter_response_rate remain the dominant signals (55% combined)
    since the JD names them directly. avg_response_time_hours and
    offer_acceptance_rate add precision: a candidate who responds within hours
    and consistently accepts offers is more reachable in practice than one
    who replies eventually and then ghosts. offer_acceptance_rate of -1 means
    no data (no prior offers) and is treated as neutral (0.5)."""
    signals = candidate.get("redrob_signals", {})

    activity_score = _activity_score(candidate, dataset_as_of)
    response_score = _clamp(float(signals.get("recruiter_response_rate", 0.0)))
    reliability_score = _clamp(float(signals.get("interview_completion_rate", 0.0)))

    rt_raw = signals.get("avg_response_time_hours")
    response_time_score = _response_time_score(float(rt_raw) if rt_raw is not None else None)

    offer_raw = float(signals.get("offer_acceptance_rate", 0.0))
    offer_acceptance = _clamp(offer_raw) if offer_raw >= 0 else 0.5

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
    """Practical fit on location, work mode, relocation, and notice period.
    Location is now 30% of this score (previously unscored despite the JD
    opening with "Location: Pune/Noida, India (Hybrid)"). Weights for work
    mode, relocation, and notice were scaled down proportionally to fund it.
    Outside-India candidates score 0 on location but can recover via
    relocation willingness and short notice. Max still sums to 1.0."""
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # Location: Pune/Noida ideal, other tier-1 Indian cities good, outside India 0.
    score += _location_score(candidate) * 0.30

    # Work mode (was 0.40, now 0.25).
    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in {"hybrid", "flexible"}:
        score += 0.25

    # Relocation willingness (was 0.30, now 0.20).
    if signals.get("willing_to_relocate"):
        score += 0.20

    # Notice period (was max 0.30, now max 0.25).
    notice = int(signals.get("notice_period_days", 180))
    if notice <= 30:
        score += 0.25
    elif notice <= 60:
        score += 0.15
    elif notice <= 90:
        score += 0.05

    return _clamp(score)


def _recruiter_demand_score(candidate: dict[str, Any], demand_caps: DemandCaps) -> float:
    """How much do recruiters already like this candidate -- independent of
    anything the candidate themselves wrote or did. saved_by_recruiters_30d,
    profile_views_received_30d, and search_appearance_30d are all signals
    *other* recruiters generated by acting on this profile, which makes them
    a kind of external, wisdom-of-the-crowd validation that the resume-text
    dimensions above can't see. Equal-weighted average of the three, each
    normalized against its own corpus-wide p95 (see DemandCaps) so a
    candidate at or above the top ~5% on a signal gets full credit on it
    rather than being scaled down by one or two extreme outliers."""
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

    # Spec's own honeypot example: "expert" proficiency claimed on multiple
    # skills with ~0 months of actual use. A single 0-duration entry could be
    # a data-entry quirk, but several together is the classic honeypot
    # signature (confirmed present as a tight cluster in this dataset: ~20
    # candidates have 3-5 such skills, everyone else has 0).
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
    """Checks for the JD's explicit "Things we do NOT want" patterns and returns
    a 0-1 risk score that multiplies down the final score, similar to honeypot_risk.
    Unlike honeypot (internally inconsistent/fake profiles), disqualifiers are about
    fit mismatches the candidate may not even be aware of.

    Three checks, each independent:
    - Consulting-only career: entire history at IT services firms (TCS, Infosys, etc.)
    - LangChain-only AI experience: LangChain in profile but no real retrieval/ranking background
    - Title-chaser pattern: 3+ roles with < 18 months tenure each
    """
    risk = 0.0
    concerns: list[str] = []
    roles = candidate.get("career_history", [])

    # Consulting-only career (only meaningful if career is at least 2 years long,
    # so a fresh graduate's first 13-month stint at TCS isn't over-penalized on
    # top of the experience-alignment penalty they already get)
    if roles:
        total_months = sum(int(r.get("duration_months") or 0) for r in roles)
        if total_months >= 24 and all(_is_consulting_company(r.get("company", "")) for r in roles):
            risk += 0.40
            concerns.append("entire career history is at IT services / consulting firms")

    # LangChain-only AI experience: LangChain in profile but without the independent
    # retrieval/ranking depth the JD requires (pre-LLM-era production experience)
    profile_text = candidate_profile_text(candidate).lower()
    if "langchain" in profile_text:
        has_retrieval_depth = any(term in profile_text for term in {
            "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
            "opensearch", "bm25", "retrieval", "ranking", "recommendation",
            "fine-tuning", "fine tuning", "ndcg", "mrr",
        })
        if not has_retrieval_depth:
            risk += 0.25
            concerns.append(
                "AI experience appears limited to LangChain / LLM-wrapper tooling "
                "without retrieval or ranking background"
            )

    # Title-chaser: 3+ short stints (< 18 months) suggests optimizing for title
    # progression over depth — the JD explicitly names this as a disqualifier
    if roles:
        short_stints = sum(1 for r in roles if int(r.get("duration_months") or 0) < 18)
        if len(roles) >= 3 and short_stints >= 3:
            risk += 0.20
            concerns.append(
                f"{short_stints} of {len(roles)} roles under 18 months — "
                "possible title-chasing pattern"
            )

    return _clamp(risk), concerns


# main function
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
    career_evidence, evidence_snippets = _career_evidence_score(candidate, idf)
    production = _production_experience_score(candidate, idf)
    behavioral = _behavioral_score(candidate, dataset_as_of)
    experience_alignment = _experience_alignment_score(candidate, jd_years_min, jd_years_max)
    logistics = _logistics_score(candidate)
    recruiter_demand = _recruiter_demand_score(candidate, demand_caps)
    honeypot_risk, concerns = _honeypot_risk(candidate)
    disqualifier_risk, disqualifier_concerns = _disqualifier_risk(candidate)

    final_score = (
        0.18 * capability_match
        + 0.18 * career_evidence
        + 0.18 * production
        + 0.18 * behavioral
        + 0.09 * experience_alignment
        + 0.09 * logistics
        + 0.10 * recruiter_demand
    )
    # Apply penalties multiplicatively so they stack without cancelling each other.
    # honeypot_risk (fake/inconsistent profiles) uses a steeper multiplier (0.65)
    # than disqualifier_risk (fit mismatches) (0.50) since fraud is worse than poor fit.
    final_score = final_score * (1.0 - 0.65 * honeypot_risk) * (1.0 - 0.50 * disqualifier_risk)

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

