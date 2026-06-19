# Architecture Notes

## 1. Feature Extraction

This is the first stage.

We do not filter candidates away early. With 100,000 candidates, lightweight scoring is cheap enough to run on everyone.

For each candidate, we extract:

- profile facts: years of experience, title, company, location
- career history evidence: descriptions and titles
- skill evidence: names, proficiency, duration
- behavioral signals: activity, responsiveness, notice period, work mode, relocation

## 2. Scoring and Ranking

We combine the features into a single ranking score.

The core categories are:

- technical/domain fit
- career evidence fit
- production experience fit
- behavioral fit
- experience alignment
- logistics fit

The most important addition is **career evidence fit**, because the JD repeatedly asks for proof of what the candidate actually built.

## 3. Honeypot Detection

The bundle warns that some profiles are impossible on purpose.

The detector checks:

- years of experience versus summed career history
- seniority implied by title versus claimed experience
- skill duration versus total experience
- behavior contradictions

The goal is not perfect fraud detection. The goal is to avoid promoting obviously suspicious profiles into the top ranks.

## 4. Reasoning

Reasoning is generated from templates, not an external LLM.

That keeps the output:

- fast
- deterministic
- easy to explain in interview
- safer against hallucination

The reasoning should quote only facts that appear in the profile or signals.

## 5. Why no separate filter stage

The previous draft had a filter stage and then a reranker.

This revision simplifies that:

- everyone gets feature extraction
- everyone gets scored
- the top 100 are selected from the full pool

That is simpler, cheaper, and easier to debug.
