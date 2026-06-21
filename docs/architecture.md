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

Seven weighted components, combined into one score and then discounted by honeypot risk:

- capability match (TF-IDF cosine similarity against the JD) -- 18%
- career evidence fit (proof of having built/shipped relevant work) -- 18%
- production experience fit (deployment/scale signals) -- 18%
- behavioral fit (Redrob activity, responsiveness, reliability) -- 18%
- experience alignment (fit to the JD's years band) -- 9%
- logistics fit (work mode, relocation, notice period) -- 9%
- recruiter demand (Redrob saves/views/search appearances) -- 10%

Capability match started at 25% and behavioral fit at 15%. Capability match was trimmed because TF-IDF cosine similarity between two short bags of words compresses into a narrow range on this dataset (tops out well under 0.2), so 25% overstated how much it could really move the score -- those points went to behavioral fit instead. When recruiter demand was added later as a 7th dimension, the other six gave up their 10% proportionally (20/20/20/20/10/10 -> 18/18/18/18/9/9) rather than cutting one dimension again.

**Capability match** is TF-IDF cosine similarity, not a fixed keyword list. The old version counted exact hits against 21 JD buzzwords, so anything outside that list (a tool, a metric) was invisible. IDF weights are learned from the 100K pool in a first streaming pass, then every candidate is scored against the JD in a second -- no network calls, no ML library, fully deterministic. See `src/tfidf.py`.

**Career evidence fit** and **production experience fit** ask a different question -- not "does this read like the JD" but "is there proof they actually built/shipped this" -- so they match against a short, fixed signal list instead of full cosine similarity (which would compress a short list into too narrow a range to matter). Each matched term is weighted by its corpus rarity, reusing the same IDF weights, so a hit on "qdrant" counts for more than a hit on "production." Career evidence fit adds one more layer on top: a regex scan for a builder verb (built, led, designed...) landing near a JD-relevant noun, e.g. "Built candidate recommendation engine." Neither word alone means much, but the pairing is a much rarer, more specific claim -- it earns a steep graded bonus (`BUILDER_PHRASE_BONUS_BY_COUNT` in `src/features.py`) and the matched phrase itself becomes the evidence shown in `matched_evidence` and the generated reasoning.

**Behavioral fit** is three equal parts: activity, response, reliability. It's built directly off the JD's own closing note: *"a candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is... not actually available. Down-weight them appropriately."* Activity decays linearly to 0 at that 180-day mark, response is `recruiter_response_rate` directly, reliability is `interview_completion_rate` directly. Recency needs an "as of" date, and using wall-clock time would make the score (and the submitted CSV) drift depending on what day the script runs -- so it's anchored to the most recent `last_active_date` seen anywhere in the pool instead, learned in that same first pass. See `extract_last_active_date` / `_activity_score` in `src/features.py`.

**Recruiter demand** is the one dimension built from what *other* recruiters did (saves, views, search appearances), not the candidate's own text or behavior -- a wisdom-of-the-crowd signal the other six dimensions can't see. Each of the three fields is normalized against its own corpus 95th percentile rather than its raw max, since these are unbounded, right-skewed counts where one or two outliers would otherwise flatten everyone else (`search_appearance_30d` maxes at 1490 vs. a p95 of 278 on this dataset). The cap is learned in the same first pass as IDF weights. See `DemandCaps` / `_recruiter_demand_score` in `src/features.py`. This doesn't conflict with the honeypot check below, which flags high saves *paired with* a near-zero response rate -- a candidate still needs to be reachable to get full credit on both.

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
