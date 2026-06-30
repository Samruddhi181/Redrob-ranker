# Architecture

1. Feature Extraction

Every candidate in the 100K pool gets scored — no early filtering. At this scale, lightweight scoring is fast enough to run on everyone, and filtering early risks quietly dropping good candidates.

For each candidate we extract:
- Profile facts — years of experience, current title, company, location
- Career history — role descriptions and titles across all past jobs
- Skills — names, proficiency level, months of use
- Behavioral signals — platform activity, recruiter response rate, notice period, work mode, relocation willingness

---

2. Scoring

Each candidate is scored on seven dimensions, then combined into a single weighted score:

| Dimension | Weight | What it measures |
|---|---|---|
| Career evidence fit | 25% | Proof of having actually built ranking/retrieval/search systems |
| Production experience | 19% | Deployment and scale signals in role descriptions |
| Behavioral fit | 18% | Platform activity, recruiter response rate, interview completion |
| Capability match | 10% | TF-IDF similarity between the candidate's profile and the JD |
| Recruiter demand | 10% | Saves, views, and search appearances from other recruiters |
| Experience alignment | 9% | Fit to the JD's 5–9 year band |
| Logistics fit | 9% | Location, work mode, notice period, relocation |

Capability match: uses TF-IDF cosine similarity rather than a fixed keyword list. IDF weights are learned from the full 100K pool in a first streaming pass, so rare JD-specific terms like "qdrant" or "ndcg" carry more weight than common words like "production" or "team." No external libraries or network calls — fully deterministic.

Career evidence fit: asks a different question than capability match: not "does this profile sound like the JD" but "did this person actually build these systems." It scans career descriptions for builder verbs (built, designed, shipped, owned...) paired with JD-relevant domain terms (ranking, retrieval, recommendation...). A bare mention of "ranking" scores lower than "built a ranking pipeline." Exposure phrases like "familiar with" or "worked with" are detected separately and capped at a much lower weight.

Behavioral fit: is built directly from the JD's own instruction: a candidate inactive for 6 months with a 5% recruiter response rate is "not actually available." Activity decays linearly to zero at the 180-day mark. The "as of" date is anchored to the most recent `last_active_date` in the pool rather than wall-clock time, so scores stay identical no matter when the script runs.

Recruiter demand: captures what other recruiters already think of this candidate — saves, profile views, and search appearances. Each field is normalized against the pool's 95th percentile rather than its raw max, since a handful of outliers would otherwise compress everyone else near zero.

---

3. Penalties

Two risk signals multiply down the final score rather than hard-filtering candidates:

**Honeypot risk** flags internally inconsistent profiles — experience vs. career history length, title seniority vs. years claimed, expert proficiency on skills with near-zero months of use. These get a steeper penalty (up to 65%) because fabricated data is worse than a poor fit.

**Disqualifier risk** flags JD-specific mismatches — consulting-only careers, LLM framework experience without retrieval depth, title-chasing patterns, candidates based outside India who aren't open to relocating. These get a softer penalty (up to 40%) since they're preferences, not hard rejections.

---

4. Reasoning

Each candidate gets a one-to-three sentence explanation generated from templates, not an LLM. This keeps reasoning fast, deterministic, and grounded — it only states facts that actually appear in the candidate's profile or platform signals, never inventing company names, project names, or specific claims.

