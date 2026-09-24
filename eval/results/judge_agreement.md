# Judge consistency — re-judging the stored RAG answers

50 answers from rag.json (source run 133e9c7) · judge=nvidia/nemotron-3-ultra-550b-a55b:free · K=3 per temperature · prompt=62bb62544847

**Repeats** judged on **openrouter** (`nvidia/nemotron-3-ultra-550b-a55b:free`); the **original** judgement in rag.json was made on **openrouter** (`nvidia/nemotron-3-ultra-550b-a55b:free`).

Same claim, context, answer, prompt and model as the original judgement; only the temperature differs. T=0.0 is production, so movement there is provider nondeterminism. `answered` is published from the verdict line for most rows; the judge decides it only for the 3 with no parsed verdict, so "would flip" counts only those.

| Metric | T=0.0 | T=0.7 |
|---|---|---|
| Rows re-judged | 50 | 50 |
| Judge calls | 150 | 150 |
| Unparseable replies | 0 | 0 |
| API errors | 0 | 0 |
| Repeat identical to original (all 3 fields) | 0.887 | 0.593 |
| `answered`: all K repeats agree | 1.000 | 0.900 |
| `answered`: Fleiss' kappa | 1.000 | 0.841 |
| `answered`: agrees with original judgement | 1.000 | 0.960 |
| `answered`: disagrees with published `answered` | 0.000 | 0.040 |
| `answered`: would flip published `answered` | 0.000 | 0.000 |
| Faithfulness: mean per-row std | 0.026 | 0.118 |
| Faithfulness: mean abs diff vs original | 0.022 | 0.111 |
| Faithfulness: Krippendorff alpha (repeats) | 0.886 | 0.506 |
| Faithfulness: Krippendorff alpha (+ original) | 0.898 | 0.531 |
| Context relevance: mean per-row std | 0.012 | 0.063 |
| Context relevance: mean abs diff vs original | 0.014 | 0.057 |
| Context relevance: Krippendorff alpha (repeats) | 0.993 | 0.929 |
| Context relevance: Krippendorff alpha (+ original) | 0.993 | 0.941 |

Kappa / alpha are n/a when undefined (all ratings identical: agreement is total but carries no information beyond that).
