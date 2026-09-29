# RAG answer quality — verdicts scored against SciFact labels, LLM-as-judge for faithfulness

300 SciFact claims (random sample, seed=13) · top_k=5 · generator=inclusionai/ling-3.0-flash-sante:free (openrouter) · judge=nvidia/nemotron-3-ultra-550b-a55b:free (openrouter)

11 answers truncated at the token budget (scored as written, with no verdict line).

## Answer quality (LLM judge)

| Metric | Score |
|---|---|
| Faithfulness (over answered) | 0.98 |
| Context relevance (all) | 0.72 |

## Claim verdicts, scored against the gold label (no judge)

The final `Verdict:` line maps SUPPORTED→SUPPORT, REFUTED→CONTRADICT, NOT ENOUGH EVIDENCE→NEI. A reply with no verdict line counts as NEI if it abstained and as `NONE` (always wrong) if it answered.

| Metric | Score |
|---|---|
| 3-class verdict accuracy | 0.78 |
| Verdict line parsed | 0.95 |
| Truncated answers (hit the token budget after 1 retry) | 11 of 300 |

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE |
|---|---|---|---|---|
| **SUPPORT** | 96 | 6 | 19 | 3 |
| **CONTRADICT** | 1 | 55 | 7 | 1 |
| **NEI** | 17 | 13 | 82 | 0 |

## Abstention — rationale oracle (headline)

`evidence` = a doc the annotators cited **with rationale sentences** was retrieved into the top-5. NEI claims have no rationale docs, so for them abstaining is the correct action. `answered` comes from the verdict line (anything but NOT ENOUGH EVIDENCE); the judge decides it only when no verdict was parsed (15 of 300 here).

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 154 | 38 |
| Abstained | 21 (false) | 87 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.58 |
| Answered (model attempted an answer) | 0.64 |
| Abstention precision (abstained & no evidence) | 0.81 |
| Abstention recall (no evidence & abstained) | 0.70 |
| False abstention (had evidence, still abstained) | 0.12 |
| Answered without evidence (hallucination risk) | 0.20 |

| Gold label | n | Answered |
|---|---|---|
| SUPPORT | 124 | 105 |
| CONTRADICT | 64 | 57 |
| NEI | 112 | 30 |

## Abstention — legacy qrels oracle

`evidence` = any BEIR qrels-relevant doc was retrieved into the top-5. BEIR marks a cited abstract relevant for NEI claims too, so this definition scores abstaining on an NEI claim as a false abstention. Kept so earlier published numbers stay traceable; same answers as above, different oracle.

| | Evidence retrieved | No evidence |
|---|---|---|
| Answered | 170 | 22 |
| Abstained | 65 (false) | 43 (correct) |

| Metric | Score |
|---|---|
| Evidence retrieved | 0.78 |
| Answered (model attempted an answer) | 0.64 |
| Abstention precision (abstained & no evidence) | 0.40 |
| Abstention recall (no evidence & abstained) | 0.66 |
| False abstention (had evidence, still abstained) | 0.28 |
| Answered without evidence (hallucination risk) | 0.11 |
