"""Blind human re-labelling of SciFact NEI disagreements — packet builder and scorer.

Offline: no LLM calls, no network. Reads a rag.json (default: the committed
eval/results/rag.json) and never writes eval/results/.

Why. In the committed run, 30 claims with gold NEI got a SUPPORT (17) or CONTRADICT (13)
verdict. SciFact's NEI means the claim's single *cited* abstract carries no rationale;
the system, though, retrieves from all 5,183 abstracts, and another passage may genuinely
decide the claim (SciFact-Open, Wadden et al. 2022, documents exactly this). Whether these
30 are model errors or label artefacts of the closed-corpus framing is an empirical
question, so a human labels each claim against exactly the passages the generator saw.

Blindness. The labeller sees only the claim and its retrieved passages — never the
model's verdict or answer, the gold label, the citations, the claim id or the doc ids —
and the 30 disagreements are mixed with controls the model got right (default 30 NEI-gold
claims predicted NEI, 20 SUPPORT/CONTRADICT-gold claims predicted correctly, 10 + 10),
sampled and shuffled under a fixed seed, so the item mix does not reveal which are which.
Item ids are opaque seeded tokens, unrelated to the claim id. The mapping lives in a
separate key file the page never references.

Passages. Rows store ``retrieved_doc_ids`` (rank order), not passage text. The generator's
context is ``prompts.context_block``: ``[n] title\\ntext`` from the hit's corpus title and
text, and SearchService hits carry the corpus text unchanged (rag_eval's own re-ask
rebuilds its passages the same way). So the packet reconstructs them from the doc ids and
the corpus — the same content, provided the corpus has not changed since the run.

Scoring. Agreement of human vs gold and human vs model, separately for the disagreements
and the controls (on a control, gold == model, so control agreement is the labeller's
calibration against the SciFact labels), Cohen's kappa, how many of the disagreements the
human sides with the model, and a SECONDARY verdict accuracy with the human labels
substituted. The headline stays on the original labels: one annotator, and an
open-corpus judgement ("do these passages decide it?") is a different task from
SciFact's closed one ("does the cited abstract decide it?").

Run:
    uv run python -m app.eval.nei_relabel build [--rag eval/results/rag.json] [--force]
        -> data/relabel/labeling.html (open it in a browser) + data/relabel/key.json
    uv run python -m app.eval.nei_relabel score --labels path/to/exported-labels.json
        -> data/relabel/report.{md,json}
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.core.paths import REPO_ROOT, RESULTS, assert_outside, display_path

LABELS = ("SUPPORT", "CONTRADICT", "NEI")
GROUPS = ("disagreement", "control_nei", "control_evidence")
SEED = 20261007
N_NEI_CONTROLS = 30
N_EVIDENCE_CONTROLS = 20
DEFAULT_RAG = RESULTS / "rag.json"
DEFAULT_OUT = REPO_ROOT / "data" / "relabel"
HTML_NAME = "labeling.html"
KEY_NAME = "key.json"
KEY_FORMAT = "nei-relabel-key/v1"
LABELS_FORMAT = "nei-relabel-labels/v1"

CAVEAT = (
    "SECONDARY, not a headline: the headline verdict accuracy stays on the original SciFact "
    "labels. One annotator (the repo owner, who also built the system — blinded to the "
    "verdicts, answers, gold labels and claim ids, but not to the study's purpose). The "
    "human judged the open-corpus question \"do these retrieved passages decide the "
    "claim?\", which is a different task from SciFact's closed-corpus label (\"does the "
    "cited abstract decide it?\"), so a relabel measures a different thing, not a "
    "corrected SciFact score."
)


class RelabelError(ValueError):
    """A packet, key, label export or rag.json that does not fit together."""


# --- selection ----------------------------------------------------------------------------


def _rng(seed: int, tag: str) -> random.Random:
    return random.Random(f"{seed}:{tag}")  # str seeds are hashed (sha512): stable across runs


def _sample(rows: Sequence[Mapping], k: int, seed: int, tag: str) -> list[Mapping]:
    pool = sorted(rows, key=lambda r: str(r["query_id"]))
    if k > len(pool):
        raise RelabelError(f"asked for {k} {tag} items, only {len(pool)} available")
    return _rng(seed, f"sample:{tag}").sample(pool, k)


def select_items(
    rows: Sequence[Mapping],
    seed: int = SEED,
    n_nei_controls: int = N_NEI_CONTROLS,
    n_evidence_controls: int = N_EVIDENCE_CONTROLS,
) -> list[dict]:
    """Key entries (shuffled, with opaque item ids) for every NEI-gold claim the model gave
    SUPPORT/CONTRADICT, plus seeded samples of controls the model got right: NEI-gold
    predicted NEI, and SUPPORT/CONTRADICT-gold predicted correctly (split evenly, any odd
    one to SUPPORT)."""
    ids = [str(r["query_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise RelabelError("rag.json has duplicate query ids")
    disagree = [r for r in rows if r["gold_label"] == "NEI" and r["predicted_label"] in ("SUPPORT", "CONTRADICT")]
    nei_ok = [r for r in rows if r["gold_label"] == "NEI" and r["predicted_label"] == "NEI"]
    n_con = n_evidence_controls // 2
    picked: list[tuple[str, Mapping]] = [("disagreement", r) for r in sorted(disagree, key=lambda r: str(r["query_id"]))]
    picked += [("control_nei", r) for r in _sample(nei_ok, n_nei_controls, seed, "control_nei")]
    for lab, k in (("SUPPORT", n_evidence_controls - n_con), ("CONTRADICT", n_con)):
        ok = [r for r in rows if r["gold_label"] == lab and r["predicted_label"] == lab]
        picked += [("control_evidence", r) for r in _sample(ok, k, seed, f"control_{lab}")]
    if not disagree:
        raise RelabelError("no NEI-gold claims with a SUPPORT/CONTRADICT prediction in this run")

    id_rng = _rng(seed, "item-ids")
    item_ids: set[str] = set()
    entries = []
    for group, r in picked:
        while (iid := f"it-{id_rng.getrandbits(40):010x}") in item_ids:
            pass
        item_ids.add(iid)
        retrieved = [str(d) for d in r.get("retrieved_doc_ids") or []]
        if not retrieved:
            raise RelabelError(f"claim {r['query_id']}: row stores no retrieved_doc_ids")
        entries.append({
            "item_id": iid,
            "claim_id": str(r["query_id"]),
            "group": group,
            "gold_label": r["gold_label"],
            "model_label": r["predicted_label"],
            "retrieved_doc_ids": retrieved,
            # Is a doc the SciFact annotators cited (qrels) among the passages shown? For a
            # NEI claim that doc has no rationale by definition — the analysis splits on it.
            "cited_doc_in_passages": bool(set(map(str, r.get("qrels_doc_ids") or [])) & set(retrieved)),
            "verdict_source": r.get("verdict_source"),
        })
    _rng(seed, "order").shuffle(entries)
    return entries


# --- packet --------------------------------------------------------------------------------


def public_items(
    entries: Sequence[Mapping], claims: Mapping[str, str], docs: Mapping[str, Mapping]
) -> list[dict]:
    """What the page carries per item: opaque id, claim text, passages (title + text, in the
    rank order the generator saw). Nothing else — no ids, labels, answers or citations."""
    out = []
    for e in entries:
        if e["claim_id"] not in claims:
            raise RelabelError(f"claim {e['claim_id']}: no claim text in the query set")
        passages = []
        for d in e["retrieved_doc_ids"]:
            if d not in docs:
                raise RelabelError(f"doc {d} (claim {e['claim_id']}): not in the corpus")
            passages.append({
                "title": (docs[d].get("title") or "").strip(),
                "text": (docs[d].get("text") or "").strip(),
            })
        out.append({"id": e["item_id"], "claim": claims[e["claim_id"]].strip(), "passages": passages})
    return out


def packet_id(items: Sequence[Mapping]) -> str:
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _json_for_script(obj: object) -> str:
    """JSON safe inside <script type="application/json">: no '<', '>' or '&' survive, so
    no passage text can close the element or open a comment."""
    s = json.dumps(obj, ensure_ascii=False)
    for ch, esc in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                    (" ", "\\u2028"), (" ", "\\u2029")):
        s = s.replace(ch, esc)
    return s


def _csp_hash(code: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(code.encode()).digest()).decode() + "'"


_STYLE = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#5f5f66;--card:#ffffff;--line:#d9d6cf;
--accent:#2357a6;--done:#2f7d4f;--claim:#f1ecdf;--warn:#9a4b00}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;--fg:#e8e6e1;--muted:#a3a19b;
--card:#1f2024;--line:#3a3b40;--accent:#8fb3f0;--done:#6cc28f;--claim:#2a2822;--warn:#f0a35e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,
"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:1.35rem;margin:.4rem 0 .6rem}
h2{font-size:1rem;margin:0}
details.guide{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.6rem .9rem;margin-bottom:1rem}
details.guide summary{cursor:pointer;font-weight:600}
.guide li{margin:.25rem 0}
.bar{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.6rem 0}
.bar .grow{flex:1 1 auto}
button{font:inherit;padding:.35rem .8rem;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
.nav{display:flex;flex-wrap:wrap;gap:4px;margin:.4rem 0 1rem}
.nav button{min-width:2.4rem;padding:.15rem .3rem;font-size:.8rem}
.nav button.done{border-color:var(--done);color:var(--done)}
.nav button.here{outline:2px solid var(--accent)}
.meta{color:var(--muted);font-size:.9rem}
.claim{background:var(--claim);border-radius:8px;padding:.8rem 1rem;font-size:1.1rem;
font-weight:600;margin:.4rem 0 1rem;overflow-wrap:anywhere}
.passage{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.7rem 1rem;margin:.6rem 0}
.passage h2{margin-bottom:.35rem;overflow-wrap:anywhere}
.passage p{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
fieldset{border:1px solid var(--line);border-radius:8px;margin:1rem 0;padding:.6rem 1rem;
background:var(--card)}
legend{font-weight:600;padding:0 .3rem}
label.choice{display:block;padding:.25rem 0;cursor:pointer}
label.pick{display:inline-block;margin-right:1rem;cursor:pointer}
textarea{width:100%;min-height:4.5rem;font:inherit;color:var(--fg);background:var(--bg);
border:1px solid var(--line);border-radius:6px;padding:.4rem}
.status{color:var(--muted);font-size:.85rem;min-height:1.2rem}
.status.warn{color:var(--warn)}
.hidden{display:none}
"""

_SCRIPT = r"""
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('packet-data').textContent);
  var ITEMS = DATA.items;
  var STORE = 'nei-relabel:' + DATA.packet_id;
  var CHOICES = [['SUPPORT', 'SUPPORT: the passages show the claim is true'],
                 ['CONTRADICT', 'CONTRADICT: the passages show the claim is false'],
                 ['NEI', 'NOT ENOUGH EVIDENCE: the passages do not decide it']];
  var VALID = {SUPPORT: 1, CONTRADICT: 1, NEI: 1};
  var KNOWN = {};
  ITEMS.forEach(function (it) { KNOWN[it.id] = it; });
  var state = loadState();
  var cur = 0;
  var noteTimer = null;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined && text !== null) { e.textContent = String(text); }
    return e;
  }
  function setStatus(msg, warn) {
    var s = $('status');
    s.textContent = msg;
    s.className = warn ? 'status warn' : 'status';
  }
  function clean(entry, it) {
    // Keep only well-formed fields for a known item.
    if (!entry || typeof entry !== 'object') { return null; }
    var out = {label: null, passages: [], note: ''};
    if (typeof entry.label === 'string' && VALID[entry.label]) { out.label = entry.label; }
    if (Array.isArray(entry.passages)) {
      entry.passages.forEach(function (n) {
        if (typeof n === 'number' && n % 1 === 0 && n >= 1 && n <= it.passages.length &&
            out.passages.indexOf(n) < 0) { out.passages.push(n); }
      });
      out.passages.sort(function (a, b) { return a - b; });
    }
    if (typeof entry.note === 'string') { out.note = entry.note.slice(0, 5000); }
    return out;
  }
  function loadState() {
    var s = {};
    try {
      var raw = window.localStorage.getItem(STORE);
      var obj = raw ? JSON.parse(raw) : null;
      var labels = obj && typeof obj === 'object' ? obj.labels : null;
      if (labels && typeof labels === 'object') {
        Object.keys(labels).forEach(function (k) {
          if (KNOWN.hasOwnProperty(k)) {
            var c = clean(labels[k], KNOWN[k]);
            if (c) { s[k] = c; }
          }
        });
      }
    } catch (e) { s = {}; }
    return s;
  }
  function save() {
    try {
      window.localStorage.setItem(STORE, JSON.stringify({labels: state, saved_at: new Date().toISOString()}));
      setStatus('Saved in this browser at ' + new Date().toLocaleTimeString() + '.', false);
    } catch (e) {
      setStatus('Could not save in browser storage here. Export your labels regularly.', true);
    }
  }
  function entryFor(id) {
    if (!state.hasOwnProperty(id)) { state[id] = {label: null, passages: [], note: ''}; }
    return state[id];
  }
  function nDone() {
    return ITEMS.filter(function (it) { return state[it.id] && state[it.id].label; }).length;
  }
  function renderNav() {
    var nav = $('nav');
    nav.textContent = '';
    ITEMS.forEach(function (it, i) {
      var b = el('button', '', String(i + 1));
      b.type = 'button';
      if (state[it.id] && state[it.id].label) { b.className = 'done'; }
      if (i === cur) { b.className += ' here'; }
      b.title = 'Item ' + (i + 1);
      b.addEventListener('click', function () { go(i); });
      nav.appendChild(b);
    });
    $('progress').textContent = nDone() + ' of ' + ITEMS.length + ' labelled';
  }
  function render() {
    var it = ITEMS[cur];
    var entry = state[it.id] || {label: null, passages: [], note: ''};
    $('pos').textContent = 'Item ' + (cur + 1) + ' of ' + ITEMS.length + ' (ref ' + it.id + ')';
    $('claim').textContent = it.claim;
    var box = $('passages');
    box.textContent = '';
    it.passages.forEach(function (p, i) {
      var card = el('section', 'passage');
      card.appendChild(el('h2', '', '[' + (i + 1) + '] ' + (p.title || '(untitled)')));
      card.appendChild(el('p', '', p.text));
      box.appendChild(card);
    });
    var choices = $('choices');
    choices.textContent = '';
    CHOICES.forEach(function (c) {
      var lab = el('label', 'choice');
      var r = el('input');
      r.type = 'radio';
      r.name = 'label';
      r.value = c[0];
      r.checked = entry.label === c[0];
      r.addEventListener('change', function () { setLabel(c[0]); });
      lab.appendChild(r);
      lab.appendChild(document.createTextNode(' ' + c[1]));
      choices.appendChild(lab);
    });
    var picks = $('picks');
    picks.textContent = '';
    it.passages.forEach(function (p, i) {
      var lab = el('label', 'pick');
      var cb = el('input');
      cb.type = 'checkbox';
      cb.value = String(i + 1);
      cb.checked = entry.passages.indexOf(i + 1) >= 0;
      cb.addEventListener('change', function () { togglePassage(i + 1, cb.checked); });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(' [' + (i + 1) + ']'));
      picks.appendChild(lab);
    });
    $('note').value = entry.note || '';
    $('prev').disabled = cur === 0;
    $('next').disabled = cur === ITEMS.length - 1;
    renderNav();
  }
  function setLabel(v) {
    entryFor(ITEMS[cur].id).label = v;
    save();
    renderNav();
  }
  function togglePassage(n, on) {
    var e = entryFor(ITEMS[cur].id);
    var at = e.passages.indexOf(n);
    if (on && at < 0) { e.passages.push(n); }
    if (!on && at >= 0) { e.passages.splice(at, 1); }
    e.passages.sort(function (a, b) { return a - b; });
    save();
  }
  function go(i) {
    flushNote();
    cur = Math.max(0, Math.min(ITEMS.length - 1, i));
    render();
    window.scrollTo(0, 0);
  }
  function flushNote() {
    if (noteTimer !== null) {
      window.clearTimeout(noteTimer);
      noteTimer = null;
      entryFor(ITEMS[cur].id).note = $('note').value.slice(0, 5000);
      save();
    }
  }
  function exportLabels() {
    flushNote();
    var labels = {};
    ITEMS.forEach(function (it) {
      if (state[it.id]) { labels[it.id] = clean(state[it.id], it); }
    });
    var out = {format: DATA.labels_format, packet_id: DATA.packet_id,
               exported_at: new Date().toISOString(), n_items: ITEMS.length,
               n_labelled: nDone(), labels: labels};
    var blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
    var url = URL.createObjectURL(blob);
    var a = el('a');
    a.href = url;
    a.download = 'labels-' + DATA.packet_id + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
    setStatus('Exported ' + nDone() + ' of ' + ITEMS.length + ' labels.', false);
  }
  function importLabels(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var obj;
      try { obj = JSON.parse(String(reader.result)); } catch (e) {
        setStatus('That file is not valid JSON.', true);
        return;
      }
      if (!obj || obj.packet_id !== DATA.packet_id || typeof obj.labels !== 'object' || !obj.labels) {
        setStatus('That file is not an export of this packet.', true);
        return;
      }
      if (!window.confirm('Replace the labels in this browser with the imported file?')) { return; }
      var s = {};
      Object.keys(obj.labels).forEach(function (k) {
        if (KNOWN.hasOwnProperty(k)) {
          var c = clean(obj.labels[k], KNOWN[k]);
          if (c) { s[k] = c; }
        }
      });
      state = s;
      save();
      render();
      setStatus('Imported ' + nDone() + ' labels.', false);
    };
    reader.readAsText(file);
  }

  $('prev').addEventListener('click', function () { go(cur - 1); });
  $('next').addEventListener('click', function () { go(cur + 1); });
  $('next-open').addEventListener('click', function () {
    for (var k = 1; k <= ITEMS.length; k++) {
      var i = (cur + k) % ITEMS.length;
      if (!(state[ITEMS[i].id] && state[ITEMS[i].id].label)) { go(i); return; }
    }
    setStatus('Every item has a label. Export when ready.', false);
  });
  $('export').addEventListener('click', exportLabels);
  $('import').addEventListener('click', function () { $('import-file').click(); });
  $('import-file').addEventListener('change', function () {
    if (this.files && this.files[0]) { importLabels(this.files[0]); }
    this.value = '';
  });
  $('note').addEventListener('input', function () {
    if (noteTimer !== null) { window.clearTimeout(noteTimer); }
    noteTimer = window.setTimeout(function () {
      noteTimer = null;
      entryFor(ITEMS[cur].id).note = $('note').value.slice(0, 5000);
      save();
    }, 400);
  });
  window.addEventListener('beforeunload', flushNote);
  document.addEventListener('keydown', function (ev) {
    var t = ev.target && ev.target.tagName;
    if (t === 'TEXTAREA' || t === 'INPUT' || ev.altKey || ev.ctrlKey || ev.metaKey) { return; }
    if (ev.key === 'ArrowLeft') { go(cur - 1); }
    else if (ev.key === 'ArrowRight') { go(cur + 1); }
    else if (ev.key === 's') { setLabel('SUPPORT'); render(); }
    else if (ev.key === 'c') { setLabel('CONTRADICT'); render(); }
    else if (ev.key === 'n') { setLabel('NEI'); render(); }
  });
  render();
  setStatus(nDone() ? 'Restored ' + nDone() + ' labels saved in this browser.' : '', false);
})();
"""

_BODY = """<main>
<h1>Claim labelling</h1>
<details class="guide" open>
<summary>Guidelines (read once before starting)</summary>
<ul>
<li>Each item shows a scientific <b>claim</b> and the <b>passages</b> retrieved for it
(paper titles and abstracts).</li>
<li>Label what the passages, <b>taken together</b>, establish about the claim:
<b>SUPPORT</b> if they show it is true; <b>CONTRADICT</b> if they show it is false;
<b>NOT ENOUGH EVIDENCE</b> if they do not decide it either way.</li>
<li>Choose NOT ENOUGH EVIDENCE when the passages are on topic but do not settle the claim:
for example a different population, species, intervention, outcome or direction of
effect, a weaker or only partial result, or a result the claim overstates.</li>
<li><b>Use only the passages shown.</b> Do not use outside knowledge and do not look
anything up, even if you think you know whether the claim is true.</li>
<li>Optional: tick the passage(s) that decided it, and add a short note
(e.g. "only shown in mice").</li>
<li>Items come from a mix; there is no expected share of any label. Judge each item on
its own.</li>
<li>Progress saves in this browser as you go. When finished (and now and then as a
backup), press <b>Export labels</b> and keep the downloaded file.
Keys: &larr;/&rarr; move, s / c / n pick a label.</li>
</ul>
</details>
<div class="bar"><span id="progress" class="meta"></span><span class="grow"></span>
<button type="button" id="next-open">Next unlabelled</button>
<button type="button" id="export">Export labels</button>
<button type="button" id="import">Import labels</button>
<input type="file" id="import-file" class="hidden" accept="application/json,.json"></div>
<div id="status" class="status" role="status"></div>
<div id="nav" class="nav"></div>
<div id="pos" class="meta"></div>
<div id="claim" class="claim"></div>
<div id="passages"></div>
<fieldset><legend>Label</legend><div id="choices"></div></fieldset>
<fieldset><legend>Which passage(s) decided it? (optional)</legend><div id="picks"></div>
</fieldset>
<fieldset><legend>Note (optional)</legend>
<textarea id="note" aria-label="Note"></textarea></fieldset>
<div class="bar"><button type="button" id="prev">&larr; Previous</button>
<span class="grow"></span><button type="button" id="next">Next &rarr;</button></div>
</main>"""


def render_html(items: Sequence[Mapping], pid: str) -> str:
    """A self-contained page: no external resources, all item text inserted with
    textContent, a CSP that allows only this page's own script and style (by hash)."""
    data = _json_for_script({"packet_id": pid, "labels_format": LABELS_FORMAT, "items": list(items)})
    csp = (
        f"default-src 'none'; script-src {_csp_hash(_SCRIPT)}; style-src {_csp_hash(_STYLE)}; "
        "img-src 'none'; connect-src 'none'; form-action 'none'; base-uri 'none'"
    )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"{csp}\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<meta name=\"referrer\" content=\"no-referrer\">\n"
        "<title>Claim labelling</title>\n"
        f"<style>{_STYLE}</style>\n</head><body>\n{_BODY}\n"
        f"<script type=\"application/json\" id=\"packet-data\">{data}</script>\n"
        f"<script>{_SCRIPT}</script>\n</body></html>\n"
    )


def build_packet(
    blob: Mapping,
    claims: Mapping[str, str],
    docs: Mapping[str, Mapping],
    seed: int = SEED,
    n_nei_controls: int = N_NEI_CONTROLS,
    n_evidence_controls: int = N_EVIDENCE_CONTROLS,
    source: str = "",
) -> tuple[str, dict]:
    """(html, key) for a rag.json blob. `claims` is {claim id: text}, `docs` is
    {doc id: {"title", "text"}} — injectable, so tests need no corpus."""
    entries = select_items(blob["rows"], seed, n_nei_controls, n_evidence_controls)
    items = public_items(entries, claims, docs)
    pid = packet_id(items)
    run = blob.get("run") or {}
    counts = {g: sum(e["group"] == g for e in entries) for g in GROUPS}
    key = {
        "format": KEY_FORMAT,
        "packet_id": pid,
        "seed": seed,
        "source": source,
        "source_run": {k: run.get(k) for k in ("git_sha", "dataset", "generator_model", "prompt_hash")},
        "n_items": len(entries),
        "counts": counts,
        "passages_note": (
            "rows store retrieved_doc_ids only; passages were rebuilt as corpus title + text "
            "in rank order, the content prompts.context_block renders for the generator"
        ),
        "items": entries,
    }
    return render_html(items, pid), key


def _guarded(path: Path) -> Path:
    return assert_outside(Path(path))


def write_packet(out_dir: Path, html: str, key: Mapping, force: bool = False) -> tuple[Path, Path]:
    out_dir = _guarded(out_dir)
    html_p, key_p = _guarded(out_dir / HTML_NAME), _guarded(out_dir / KEY_NAME)
    if key_p.exists() and not force:
        try:
            old = json.loads(key_p.read_text()).get("packet_id")
        except (OSError, ValueError, AttributeError):
            old = None
        if old != key["packet_id"]:
            raise SystemExit(
                f"{key_p} holds a different packet ({old}); labels exported for it would no "
                "longer match. Move it aside or pass --force."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    html_p.write_text(html, encoding="utf-8")
    key_p.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return html_p, key_p


# --- scoring -------------------------------------------------------------------------------

_ALIASES = {
    "SUPPORT": "SUPPORT", "SUPPORTS": "SUPPORT", "SUPPORTED": "SUPPORT",
    "CONTRADICT": "CONTRADICT", "CONTRADICTS": "CONTRADICT", "REFUTED": "CONTRADICT",
    "REFUTE": "CONTRADICT", "REFUTES": "CONTRADICT",
    "NEI": "NEI", "NOT ENOUGH EVIDENCE": "NEI", "NOT ENOUGH INFO": "NEI", "NOINFO": "NEI",
}


def normalize_label(v: object) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise RelabelError(f"label {v!r} is not a string")
    s = " ".join(v.replace("_", " ").split()).upper()
    if not s:
        return None
    if s not in _ALIASES:
        raise RelabelError(f"unknown label {v!r}")
    return _ALIASES[s]


def load_labels(export: Mapping, key: Mapping) -> dict[str, dict]:
    """{item id: {label, passages, note}} for the LABELLED items of an export of this
    packet. A different packet, an unknown item id or an unknown label is refused."""
    if export.get("packet_id") != key.get("packet_id"):
        raise RelabelError(
            f"labels are for packet {export.get('packet_id')!r}, key is {key.get('packet_id')!r}"
        )
    raw = export.get("labels")
    if not isinstance(raw, Mapping):
        raise RelabelError("export has no 'labels' object")
    known = {e["item_id"] for e in key["items"]}
    if unknown := sorted(set(raw) - known):
        raise RelabelError(f"{len(unknown)} labelled item ids are not in the key: {unknown[:5]}")
    out = {}
    for iid, v in raw.items():
        if not isinstance(v, Mapping):
            raise RelabelError(f"item {iid}: entry is not an object")
        lab = normalize_label(v.get("label"))
        if lab is None:
            continue
        passages = v.get("passages") or []
        note = v.get("note") or ""
        out[iid] = {
            "label": lab,
            "passages": [int(p) for p in passages if isinstance(p, int) and not isinstance(p, bool)],
            "note": note if isinstance(note, str) else "",
        }
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def cohen_kappa(a: Sequence[str], b: Sequence[str], cats: Sequence[str] = LABELS) -> float | None:
    """Cohen's kappa over `cats`; None when it is undefined or uninformative: no pairs, or
    a rater who uses a single category throughout (then kappa is <= 0 whatever the other
    rater does — gold is NEI on every disagreement item, for instance)."""
    if len(a) != len(b):
        raise ValueError("rater sequences differ in length")
    n = len(a)
    if n == 0 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return None
    return round((po - pe) / (1 - pe), 4)


def _agreement(pairs: Sequence[tuple[str, str]]) -> dict:
    k = sum(x == y for x, y in pairs)
    n = len(pairs)
    return {"n": n, "agree": k, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def _confusion(pairs: Sequence[tuple[str, str]]) -> dict:
    """{row label: {human label: count}} for (row, human) pairs."""
    return {r: {h: sum(p == (r, h) for p in pairs) for h in LABELS} for r in LABELS}


def _accuracy(rows: Sequence[Mapping], gold: Mapping[str, str]) -> dict:
    k = sum(r["predicted_label"] == gold[str(r["query_id"])] for r in rows)
    n = len(rows)
    return {"n": n, "correct": k, "verdict_accuracy": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def check_rows(key: Mapping, rows: Sequence[Mapping]) -> dict[str, Mapping]:
    """{claim id: row}, refusing a rag.json whose rows no longer match the key (a
    regenerated run would silently change what the labels are compared against)."""
    by_id = {str(r["query_id"]): r for r in rows}
    for e in key["items"]:
        r = by_id.get(e["claim_id"])
        if r is None:
            raise RelabelError(f"claim {e['claim_id']} ({e['item_id']}) is not in this rag.json")
        got = (r["gold_label"], r["predicted_label"], [str(d) for d in r.get("retrieved_doc_ids") or []])
        want = (e["gold_label"], e["model_label"], e["retrieved_doc_ids"])
        if got != want:
            raise RelabelError(
                f"claim {e['claim_id']}: rag.json row (gold, prediction, passages) differs from "
                "the key — the run was regenerated after the packet was built"
            )
    return by_id


def score(key: Mapping, labels: Mapping[str, Mapping], blob: Mapping) -> dict:
    rows = blob["rows"]
    check_rows(key, rows)
    items = key["items"]
    lab = {e["item_id"]: labels[e["item_id"]]["label"] for e in items if e["item_id"] in labels}

    def group(*names: str) -> list[Mapping]:
        return [e for e in items if e["group"] in names and e["item_id"] in lab]

    def block(es: Sequence[Mapping]) -> dict:
        hg = [(e["gold_label"], lab[e["item_id"]]) for e in es]
        hm = [(e["model_label"], lab[e["item_id"]]) for e in es]
        return {
            "n": len(es),
            "human_vs_gold": _agreement(hg),
            "human_vs_model": _agreement(hm),
            "kappa_human_vs_gold": cohen_kappa([g for g, _ in hg], [h for _, h in hg]),
            "kappa_human_vs_model": cohen_kappa([m for m, _ in hm], [h for _, h in hm]),
            "confusion_gold_x_human": _confusion(hg),
            "confusion_model_x_human": _confusion(hm),
        }

    dis = group("disagreement")
    sides_model = [e for e in dis if lab[e["item_id"]] == e["model_label"]]
    sides_gold = [e for e in dis if lab[e["item_id"]] == "NEI"]
    opposite = [e for e in dis if e not in sides_model and e not in sides_gold]

    def split(es: Sequence[Mapping]) -> dict:
        return {
            "cited_doc_in_passages": sum(e["cited_doc_in_passages"] for e in es),
            "cited_doc_not_in_passages": sum(not e["cited_doc_in_passages"] for e in es),
        }

    gold = {str(r["query_id"]): r["gold_label"] for r in rows}
    relabel_dis = {**gold, **{e["claim_id"]: lab[e["item_id"]] for e in dis}}
    relabel_all = {**gold, **{e["claim_id"]: lab[e["item_id"]] for e in items if e["item_id"] in lab}}
    original = _accuracy(rows, gold)
    stored = blob.get("verdict_accuracy")
    n_total = {g: sum(e["group"] == g for e in items) for g in GROUPS}

    return {
        "packet_id": key["packet_id"],
        "source_run": key.get("source_run"),
        "n_items": len(items),
        "n_labelled": len(lab),
        "labelled_by_group": {g: len(group(g)) for g in GROUPS},
        "items_by_group": n_total,
        "disagreement": block(dis),
        "controls": block(group("control_nei", "control_evidence")),
        "control_nei": block(group("control_nei")),
        "control_evidence": block(group("control_evidence")),
        "all_items": block(group(*GROUPS)),
        "disagreement_outcome": {
            "n_labelled": len(dis),
            "n_total": n_total["disagreement"],
            "human_sides_with_model": len(sides_model),
            "human_sides_with_gold_nei": len(sides_gold),
            "human_opposite_stance": len(opposite),
            "sides_with_model_by_model_label": {
                m: sum(e["model_label"] == m for e in sides_model) for m in ("SUPPORT", "CONTRADICT")
            },
            "sides_with_model_split": split(sides_model),
            "all_split": split(dis),
        },
        "secondary_verdict_accuracy": {
            "caveat": CAVEAT,
            "original_labels": original,
            "stored_verdict_accuracy": stored,
            "stored_matches": stored is None or abs(original["verdict_accuracy"] - stored) < 5e-4,
            "human_relabels_disagreements_only": {
                **_accuracy(rows, relabel_dis),
                "labels_changed": sum(relabel_dis[q] != gold[q] for q in gold),
                "note": "one-sided: only the NEI-gold disagreements are re-judged, so this can "
                        "only rise; it is not an estimate of accuracy under human labels",
            },
            "human_relabels_all_sampled_items": {
                **_accuracy(rows, relabel_all),
                "labels_changed": sum(relabel_all[q] != gold[q] for q in gold),
                "note": "two-sided on the sample: controls the human labels differently from "
                        "gold now count against the model too; the other claims keep gold",
            },
        },
        "per_disagreement": [
            {
                "item_id": e["item_id"], "claim_id": e["claim_id"], "gold": e["gold_label"],
                "model": e["model_label"], "human": lab.get(e["item_id"]),
                "decided_by": (labels.get(e["item_id"]) or {}).get("passages", []),
                "cited_doc_in_passages": e["cited_doc_in_passages"],
                "note": (labels.get(e["item_id"]) or {}).get("note", ""),
            }
            for e in sorted((e for e in items if e["group"] == "disagreement"), key=lambda e: e["claim_id"])
        ],
    }


def _f(v: object) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _agree_md(a: Mapping) -> str:
    ci = a["ci95"]
    ci_s = f" (95% CI {ci[0]:.3f}–{ci[1]:.3f})" if ci else ""
    return f"{a['agree']}/{a['n']} = {_f(a['rate'])}{ci_s}"


def _cell(s: object) -> str:
    return " ".join(str(s).split()).replace("|", "\\|")


def to_markdown(rep: Mapping) -> str:
    d, sec = rep["disagreement_outcome"], rep["secondary_verdict_accuracy"]
    lines = [
        "# Blind human re-labelling of SciFact NEI disagreements",
        "",
        f"Packet `{rep['packet_id']}`; source run git_sha `{(rep.get('source_run') or {}).get('git_sha')}`. "
        f"Labelled {rep['n_labelled']}/{rep['n_items']} items "
        + "(" + ", ".join(f"{g} {rep['labelled_by_group'][g]}/{rep['items_by_group'][g]}" for g in GROUPS) + ").",
        "",
        "The labeller saw each claim with the 5 passages the generator saw, blind to the model's "
        "verdict and answer, the gold label and the claim id. Controls are claims the model got "
        "right, so on a control gold = model and agreement there is the labeller's calibration "
        "against SciFact's labels.",
        "",
        "## Agreement",
        "",
        "| Items | n | human vs gold | human vs model | kappa (gold) | kappa (model) |",
        "|---|---:|---|---|---:|---:|",
    ]
    for name, k in (("Disagreements (gold NEI, model S/C)", "disagreement"), ("Controls (all)", "controls"),
                    ("Controls: NEI-gold, model NEI", "control_nei"),
                    ("Controls: S/C-gold, model right", "control_evidence"), ("All items", "all_items")):
        b = rep[k]
        lines.append(
            f"| {name} | {b['n']} | {_agree_md(b['human_vs_gold'])} | {_agree_md(b['human_vs_model'])} "
            f"| {_f(b['kappa_human_vs_gold'])} | {_f(b['kappa_human_vs_model'])} |"
        )
    lines += [
        "",
        "Kappa is n/a where it is undefined (one rater uses a single label throughout — gold is "
        "NEI on every disagreement item).",
        "",
        "## The disagreements",
        "",
        f"Of {d['n_labelled']} labelled disagreements ({d['n_total']} in the packet), the human "
        f"sides with the **model** on {d['human_sides_with_model']} "
        f"(SUPPORT {d['sides_with_model_by_model_label']['SUPPORT']}, CONTRADICT "
        f"{d['sides_with_model_by_model_label']['CONTRADICT']}), with the **gold NEI** on "
        f"{d['human_sides_with_gold_nei']}, and picks the opposite stance on {d['human_opposite_stance']}.",
        "",
        f"Sided with the model: {d['sides_with_model_split']['cited_doc_in_passages']} had the "
        f"SciFact-cited (no-rationale) abstract among the passages, "
        f"{d['sides_with_model_split']['cited_doc_not_in_passages']} did not "
        f"(all labelled disagreements: {d['all_split']['cited_doc_in_passages']} / "
        f"{d['all_split']['cited_doc_not_in_passages']}).",
        "",
        "Human × gold on controls (rows gold, columns human):",
        "",
        "| gold \\ human | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * len(LABELS),
    ]
    conf = rep["controls"]["confusion_gold_x_human"]
    lines += [f"| {g} | " + " | ".join(str(conf[g][h]) for h in LABELS) + " |" for g in LABELS]
    lines += [
        "",
        "## Secondary: verdict accuracy under human relabels",
        "",
        f"> {sec['caveat']}",
        "",
        "| Labels | correct / n | verdict accuracy (95% Wilson CI) | labels changed |",
        "|---|---:|---|---:|",
    ]
    for name, k in (("Original SciFact labels (headline)", "original_labels"),
                    ("Human relabels of the NEI disagreements only", "human_relabels_disagreements_only"),
                    ("Human relabels of every sampled item", "human_relabels_all_sampled_items")):
        a = sec[k]
        ci = a["ci95"]
        lines.append(
            f"| {name} | {a['correct']}/{a['n']} | {_f(a['verdict_accuracy'])}"
            f" ({ci[0]:.3f}–{ci[1]:.3f}) | {a.get('labels_changed', 0)} |"
        )
    lines += [
        "",
        f"- Disagreements only: {sec['human_relabels_disagreements_only']['note']}.",
        f"- Every sampled item: {sec['human_relabels_all_sampled_items']['note']}.",
    ]
    if not sec["stored_matches"]:
        lines.append(
            f"- WARNING: recomputed original accuracy {sec['original_labels']['verdict_accuracy']} "
            f"differs from the run's stored {sec['stored_verdict_accuracy']}."
        )
    lines += [
        "",
        "## Per disagreement item",
        "",
        "| item | claim | gold | model | human | decided by | cited doc shown | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in rep["per_disagreement"]:
        lines.append(
            f"| {p['item_id']} | {p['claim_id']} | {p['gold']} | {p['model']} | {p['human'] or '—'} | "
            f"{','.join(map(str, p['decided_by'])) or '—'} | {'yes' if p['cited_doc_in_passages'] else 'no'} "
            f"| {_cell(p['note'])} |"
        )
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, rep: Mapping) -> tuple[Path, Path]:
    out_dir = _guarded(out_dir)
    md_p, js_p = _guarded(out_dir / "report.md"), _guarded(out_dir / "report.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_p.write_text(to_markdown(rep), encoding="utf-8")
    js_p.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return md_p, js_p


# --- CLI ----------------------------------------------------------------------------------


def _load_corpus(dataset: str) -> tuple[dict[str, str], dict[str, dict]]:
    from app.ingest.corpus import load_documents, load_queries_qrels  # heavy: CLI only

    claims, _ = load_queries_qrels(dataset)
    docs = {d["doc_id"]: d for d in load_documents()}
    return claims, docs


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.eval.nei_relabel", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the blind labelling page + key to data/relabel/")
    b.add_argument("--rag", type=Path, default=DEFAULT_RAG)
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--n-nei-controls", type=int, default=N_NEI_CONTROLS)
    b.add_argument("--n-evidence-controls", type=int, default=N_EVIDENCE_CONTROLS)
    b.add_argument("--force", action="store_true", help="overwrite a key for a different packet")
    s = sub.add_parser("score", help="score exported labels against the key")
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--key", type=Path, default=None, help="default: <out>/key.json")
    s.add_argument("--rag", type=Path, default=DEFAULT_RAG)
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    _guarded(args.out)
    blob = json.loads(args.rag.read_text())
    try:
        if args.cmd == "build":
            dataset = (blob.get("run") or {}).get("dataset") or "beir/scifact/test"
            claims, docs = _load_corpus(dataset)
            html, key = build_packet(blob, claims, docs, args.seed, args.n_nei_controls,
                                     args.n_evidence_controls, source=display_path(args.rag.resolve()))
            html_p, key_p = write_packet(args.out, html, key, force=args.force)
            print(f"packet {key['packet_id']}: {key['n_items']} items {key['counts']}")
            print(f"labelling page: {html_p}")
            print(f"key (do not open while labelling): {key_p}")
        else:
            key = json.loads((args.key or args.out / KEY_NAME).read_text())
            labels = load_labels(json.loads(args.labels.read_text()), key)
            rep = score(key, labels, blob)
            md_p, js_p = write_report(args.out, rep)
            d = rep["disagreement_outcome"]
            print(f"labelled {rep['n_labelled']}/{rep['n_items']}; human sides with the model on "
                  f"{d['human_sides_with_model']}/{d['n_labelled']} labelled disagreements")
            print(f"report: {md_p}\n        {js_p}")
    except RelabelError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
