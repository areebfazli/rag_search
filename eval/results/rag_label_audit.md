# RAG verdicts under the SciFact label audit (secondary artifact)

Source run: `eval/results/rag.json` (git aa98231, generator `inclusionai/ling-3.0-flash-sante:free`). Same stored answers, re-scored offline — no LLM calls.

**The headline stays the original SciFact labels** (`eval/results/rag.md`). This page is a sensitivity check against one published label audit, whose corrections come from a **single annotator** and have not been independently ratified; the audit's own authors recommend a multi-annotator re-audit before any corrected release.

Audit: Sylvestre, J. (2026). Gold Label Errors in the SciFact Benchmark: An LLM-Assisted Annotation Audit. BioNLP 2026. https://aclanthology.org/2026.bionlp-1.9/ Data: https://github.com/Kefez/scifact-audit-bionlp2026 @ `5711f3fdf2305552df2b6e700dca2ebd7c8806e1` — `corrections/dev_corrections.json` (corrections), `results/stage2_manual_review/scifact_false_alarms_25_review.csv` + `results/stage2_manual_review/scifact_false_alarms_remaining_24_review.csv` (debatable ids, cross-checked against `paper/scifact_audit_bionlp2026.tex` Appendix A). License: corrections + review CSVs: CC BY 4.0 (derived from SciFact); code: MIT.

| Labels | n | Verdict accuracy | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|---|---|
| Original SciFact labels (headline) | 300 | 0.7967 | 0.8145 (n=124) | 0.8906 (n=64) | 0.7232 (n=112) |
| Corrected labels — strict (the 11 confirmed errors only) | 300 | 0.8300 | 0.8793 (n=116) | 0.9242 (n=66) | 0.7288 (n=118) |
| Corrected labels, 8 debatable claims excluded from scoring | 292 | 0.8356 | 0.8909 (n=110) | 0.9375 (n=64) | 0.7288 (n=118) |

11 of 300 claims change label under the strict corrections; on those, the stored prediction goes wrong→right on 10 and right→wrong on 0. Accuracy moves 0.7967 → 0.8300 (strict) and → 0.8356 with the 8 debatable claims excluded (n=292). The predictions are identical in every row; only the answer key differs, so this is not a paired test of a model change (to compare two runs under the corrected key: `make rag-compare A=… B=… ARGS=--labels=audit`).

## Caveats — read before quoting the corrected numbers

- **Single annotator.** Every correction and every debatable call is one person's judgement (with an LLM second opinion in chat), not an adjudicated re-annotation.
- **LLM-assisted.** 8 of the 11 errors come from the 57 pairs an LLM screen (GPT-5.4-mini) flagged, adjudicated with a frontier-LLM (GPT-5.4) second opinion; the other 3 from the same annotator's review of the 152 unflagged pairs (paper, Stage 2). The corrections therefore lean toward how an LLM reads the evidence, so an LLM generator agreeing with them is expected in part — read the gain as a one-sided sensitivity check, not as hidden accuracy.
- **One direction only.** Only the 188 evidence-bearing claims (209 pairs) were audited; the 112 NEI claims were not, so a label can move to NEI or flip, but an NEI claim can never be corrected to SUPPORT/CONTRADICT.
- **Not a model comparison.** Same predictions, different key: the delta measures the labels, not the pipeline. Compare pipelines under one key (`rag_compare --labels`).

## How corrections map onto claim labels

The audit corrects claim–doc pairs; our label is per claim (`load_claim_labels`: NEI if no doc has rationale, else the one label all rationale docs share). A correction relabels its doc (`doc_id: null` = every rationale doc of the claim); a doc corrected to NEI drops out of the claim's rationale set; the claim label is then re-derived (none left → NEI; mixed SUPPORT/CONTRADICT → refused). All 11 strict corrections are `doc_id: null` on single-rationale-doc claims, so each maps 1:1. A claim relabelled NEI has no rationale doc, so its `evidence` flag becomes false and abstaining becomes correct. Qrels, retrieval and the qrels oracle are unchanged.

## Claims whose label changed (strict)

| Claim | Error type | Gold before → after | Predicted | Correct before → after | Evidence before → after | Abstention class before → after |
|---|---|---|---|---|---|---|
| 208 | overgeneralization | SUPPORT → CONTRADICT | CONTRADICT | no → yes | yes → yes | answered_with_evidence → answered_with_evidence |
| 343 | outcome_mismatch | SUPPORT → NEI | NONE | no → no | yes → no | false_abstention → correct_abstention |
| 593 | label_reversal | SUPPORT → CONTRADICT | CONTRADICT | no → yes | yes → yes | answered_with_evidence → answered_with_evidence |
| 770 | outcome_mismatch | SUPPORT → NEI | NEI | no → yes | yes → no | false_abstention → correct_abstention |
| 808 | label_reversal | SUPPORT → CONTRADICT | CONTRADICT | no → yes | yes → yes | answered_with_evidence → answered_with_evidence |
| 847 | entity_mismatch | CONTRADICT → NEI | NEI | no → yes | yes → no | false_abstention → correct_abstention |
| 879 | label_reversal | CONTRADICT → SUPPORT | SUPPORT | no → yes | yes → yes | answered_with_evidence → answered_with_evidence |
| 1216 | entity_mismatch | SUPPORT → NEI | NEI | no → yes | yes → no | false_abstention → correct_abstention |
| 1274 | entity_mismatch | SUPPORT → NEI | NEI | no → yes | yes → no | false_abstention → correct_abstention |
| 1368 | outcome_mismatch | SUPPORT → NEI | NEI | no → yes | no → no | correct_abstention → correct_abstention |
| 1385 | label_reversal | SUPPORT → CONTRADICT | CONTRADICT | no → yes | yes → yes | answered_with_evidence → answered_with_evidence |

## Debatable claims (kept at their gold label; excluded in the last tier)

8 of the audit's 8 debatable claims (57, 163, 431, 540, 982, 1041, 1132, 1150) are in this run.

| Claim | Gold | Predicted | Correct |
|---|---|---|---|
| 57 | CONTRADICT | CONTRADICT | yes |
| 163 | SUPPORT | SUPPORT | yes |
| 431 | SUPPORT | NEI | no |
| 540 | SUPPORT | SUPPORT | yes |
| 982 | SUPPORT | SUPPORT | yes |
| 1041 | CONTRADICT | NEI | no |
| 1132 | SUPPORT | NEI | no |
| 1150 | SUPPORT | SUPPORT | yes |

## Per-document error (reported, not applied)

Claim 597: the corrections file records `doc_id None -> NEI`, but the paper excludes it from the 11 and describes one of the claim's evidence docs (12779444, mortality not incidence) as mismatched. Applied per document, the claim would stay SUPPORT (rationale docs 3 -> 2); this run's evidence flag would not change.

## Tier detail

### Original SciFact labels (headline)

n = 300 · verdict accuracy **0.7967** (239/300; 95% Wilson CI 0.748–0.838)

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE | n | Accuracy |
|---|---|---|---|---|---|---|
| **SUPPORT** | 101 | 6 | 16 | 1 | 124 | 0.8145 |
| **CONTRADICT** | 1 | 57 | 6 | 0 | 64 | 0.8906 |
| **NEI** | 18 | 13 | 81 | 0 | 112 | 0.7232 |

Abstention (rationale oracle): answered 0.6533 · evidence retrieved 0.5833 · abstention precision 0.8269 · recall 0.6880 · false abstention 0.1029 · answered without evidence 0.1990 · quadrants (ans+ev / ans−ev / false abst. / correct abst.) 157 / 39 / 18 / 86

### Corrected labels — strict (the 11 confirmed errors only)

n = 300 · verdict accuracy **0.8300** (249/300; 95% Wilson CI 0.783–0.868)

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE | n | Accuracy |
|---|---|---|---|---|---|---|
| **SUPPORT** | 102 | 2 | 12 | 0 | 116 | 0.8793 |
| **CONTRADICT** | 0 | 61 | 5 | 0 | 66 | 0.9242 |
| **NEI** | 18 | 13 | 86 | 1 | 118 | 0.7288 |

Abstention (rationale oracle): answered 0.6533 · evidence retrieved 0.5667 · abstention precision 0.8750 · recall 0.7000 · false abstention 0.0765 · answered without evidence 0.1990 · quadrants (ans+ev / ans−ev / false abst. / correct abst.) 157 / 39 / 13 / 91

### Corrected labels, 8 debatable claims excluded from scoring

n = 292 · verdict accuracy **0.8356** (244/292; 95% Wilson CI 0.789–0.874)

| Gold \ predicted | SUPPORT | CONTRADICT | NEI | NONE | n | Accuracy |
|---|---|---|---|---|---|---|
| **SUPPORT** | 98 | 2 | 10 | 0 | 110 | 0.8909 |
| **CONTRADICT** | 0 | 60 | 4 | 0 | 64 | 0.9375 |
| **NEI** | 18 | 13 | 86 | 1 | 118 | 0.7288 |

Abstention (rationale oracle): answered 0.6541 · evidence retrieved 0.5651 · abstention precision 0.8713 · recall 0.6929 · false abstention 0.0788 · answered without evidence 0.2042 · quadrants (ans+ev / ans−ev / false abst. / correct abst.) 152 / 39 / 13 / 88
