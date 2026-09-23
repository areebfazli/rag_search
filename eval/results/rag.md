# RAG answer quality — verdicts scored against SciFact labels, LLM-as-judge for faithfulness

50 SciFact claims (random sample, seed=13) · top_k=5 · generator=openai/gpt-oss-120b · judge=qwen/qwen3.8-27b

## Answer quality (LLM judge)

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.98 |
| Context relevance (all) | 0.70 |

## Claim verdicts, scored against the gold label (no judge)

The final `Verdict:` line maps SUPPORTED→SUPPORT, REFUTED→CONTRADICT, NOT ENOUGH EVIDENCE→NEI. A reply with no verdict line counts as NEI if it abstained and as `NONE` (always wrong) if it answered.

| Metric | Score |
|---|---|
| 3-class verdict accuracy | 0.70 |
| Verdict line parsed | 0.86 |

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** | 12 | 1 | 2 | 4 |
| **CONTRADICT** | 0 | 9 | 4 | 0 |
| **NEI** | 4 | 0 | 14 | 0 |

## Abstention — rationale oracle (headline)

`evidence` = a doc the annotators cited **with rationale sentences** was retrieved into the top-5. NEI claims have no rationale docs, so for them abstaining is the correct action. `answered` comes from the verdict line (anything but NOT ENOUGH EVIDENCE); the judge decides it only when no verdict was parsed (7 of 50 here).

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 25 | 5 |
| Abstained | 4 (false) | 16 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.58 |
| Answered (model attempted an answer) | 0.60 |
| Abstention precision (abstained & no evidence) | 0.80 |
| Abstention recall (no evidence & abstained) | 0.76 |
| False abstention (had evidence, still abstained) | 0.14 |
| Answered without evidence (hallucination risk) | 0.17 |

| Gold label | n | Answered |
|---|---|---|
| SUPPORT | 19 | 17 |
| CONTRADICT | 13 | 9 |
| NEI | 18 | 4 |

## Abstention — legacy qrels oracle

`evidence` = any BEIR qrels-relevant doc was retrieved into the top-5. BEIR marks a cited abstract relevant for NEI claims too, so this definition scores abstaining on an NEI claim as a false abstention. Kept so earlier published numbers stay traceable; same answers as above, different oracle.

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 28 | 2 |
| Abstained | 13 (false) | 7 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.82 |
| Answered (model attempted an answer) | 0.60 |
| Abstention precision (abstained & no evidence) | 0.35 |
| Abstention recall (no evidence & abstained) | 0.78 |
| False abstention (had evidence, still abstained) | 0.32 |
| Answered without evidence (hallucination risk) | 0.07 |
