# RAG answer quality — verdicts scored against SciFact labels, LLM-as-judge for faithfulness

300 SciFact claims (random sample, seed=13) · top_k=5 · generator=ministral-14b-latest (mistral) · judge=nvidia/nemotron-3-ultra-550b-a55b:free (openrouter)

## Answer quality (LLM judge)

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.98 |
| Context relevance (all) | 0.69 |

## Claim verdicts, scored against the gold label (no judge)

The final `Verdict:` line maps SUPPORTED→SUPPORT, REFUTED→CONTRADICT, NOT ENOUGH EVIDENCE→NEI. A reply with no verdict line counts as NEI if it is complete and abstained, and as `NONE` (always wrong) if it answered or was cut off at the token budget / empty.

| Metric | Score |
|---|---|
| 3-class verdict accuracy | 0.76 |
| Verdict line parsed | 1.00 |
| Truncated answers (hit the token budget after 1 retry) | 0 of 300 |
| Re-asked (claim with no verdict: one verdict-only call) | 2 of 300 (2 gave a verdict) |

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** | 91 | 4 | 29 | 0 |
| **CONTRADICT** | 2 | 48 | 14 | 0 |
| **NEI** | 10 | 14 | 88 | 0 |

## Abstention — rationale oracle (headline)

`evidence` = a doc the annotators cited **with rationale sentences** was retrieved into the top-5. NEI claims have no rationale docs, so for them abstaining is the correct action. `answered` comes from the verdict line (anything but NOT ENOUGH EVIDENCE); the judge decides it only when no verdict was parsed (0 of 300 here).

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 140 | 29 |
| Abstained | 35 (false) | 96 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.58 |
| Answered (model attempted an answer) | 0.56 |
| Abstention precision (abstained & no evidence) | 0.73 |
| Abstention recall (no evidence & abstained) | 0.77 |
| False abstention (had evidence, still abstained) | 0.20 |
| Answered without evidence (hallucination risk) | 0.17 |

| Gold label | n | Answered |
|---|---|---|
| SUPPORT | 124 | 95 |
| CONTRADICT | 64 | 50 |
| NEI | 112 | 24 |

## Abstention — legacy qrels oracle

`evidence` = any BEIR qrels-relevant doc was retrieved into the top-5. BEIR marks a cited abstract relevant for NEI claims too, so this definition scores abstaining on an NEI claim as a false abstention. Kept so earlier published numbers stay traceable; same answers as above, different oracle.

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 154 | 15 |
| Abstained | 81 (false) | 50 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.78 |
| Answered (model attempted an answer) | 0.56 |
| Abstention precision (abstained & no evidence) | 0.38 |
| Abstention recall (no evidence & abstained) | 0.77 |
| False abstention (had evidence, still abstained) | 0.34 |
| Answered without evidence (hallucination risk) | 0.09 |
