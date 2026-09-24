# RAG answer quality — verdicts scored against SciFact labels, LLM-as-judge for faithfulness

50 SciFact claims (random sample, seed=13) · top_k=5 · generator=openai/gpt-oss-120b (openrouter) · judge=nvidia/nemotron-3-ultra-550b-a55b:free (openrouter)

## Answer quality (LLM judge)

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.83 |
| Context relevance (all) | 0.61 |

## Claim verdicts, scored against the gold label (no judge)

The final `Verdict:` line maps SUPPORTED→SUPPORT, REFUTED→CONTRADICT, NOT ENOUGH EVIDENCE→NEI. A reply with no verdict line counts as NEI if it abstained and as `NONE` (always wrong) if it answered.

| Metric | Score |
|---|---|
| 3-class verdict accuracy | 0.78 |
| Verdict line parsed | 0.94 |
| Truncated answers (hit the token budget after 1 retry) | 0 of 50 |

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** | 14 | 1 | 2 | 2 |
| **CONTRADICT** | 0 | 11 | 1 | 1 |
| **NEI** | 4 | 0 | 14 | 0 |

## Abstention — rationale oracle (headline)

`evidence` = a doc the annotators cited **with rationale sentences** was retrieved into the top-5. NEI claims have no rationale docs, so for them abstaining is the correct action. `answered` comes from the verdict line (anything but NOT ENOUGH EVIDENCE); the judge decides it only when no verdict was parsed (3 of 50 here).

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 27 | 6 |
| Abstained | 2 (false) | 15 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.58 |
| Answered (model attempted an answer) | 0.66 |
| Abstention precision (abstained & no evidence) | 0.88 |
| Abstention recall (no evidence & abstained) | 0.71 |
| False abstention (had evidence, still abstained) | 0.07 |
| Answered without evidence (hallucination risk) | 0.18 |

| Gold label | n | Answered |
|---|---|---|
| SUPPORT | 19 | 17 |
| CONTRADICT | 13 | 12 |
| NEI | 18 | 4 |

## Abstention — legacy qrels oracle

`evidence` = any BEIR qrels-relevant doc was retrieved into the top-5. BEIR marks a cited abstract relevant for NEI claims too, so this definition scores abstaining on an NEI claim as a false abstention. Kept so earlier published numbers stay traceable; same answers as above, different oracle.

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 30 | 3 |
| Abstained | 11 (false) | 6 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.82 |
| Answered (model attempted an answer) | 0.66 |
| Abstention precision (abstained & no evidence) | 0.35 |
| Abstention recall (no evidence & abstained) | 0.67 |
| False abstention (had evidence, still abstained) | 0.27 |
| Answered without evidence (hallucination risk) | 0.09 |
