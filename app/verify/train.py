"""Fine-tune PubMedBERT as a 3-way evidence verifier on beir/scifact/train — CPU, resumable.

Data: app.verify.train_data (the Ling-answered 100 train claims and every claim sharing
a doc with them are held out; validation is grouped by cited doc). Model:
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (MIT) with a fresh
3-way classification head, fed [CLS] claim [SEP] title. abstract [SEP] (with BERT's
segment ids, exactly as the tokenizer builds a pair at inference), the abstract truncated
to `max_length`.

The same training runs on a free Kaggle GPU (the usual route; see data/kaggle/README.md):
`python -m app.verify.kaggle_export` writes these exact pairs plus a self-contained
notebook, and the model it returns is unzipped into data/models/verifier/model/.

Optimisation: AdamW (lr 2e-5, weight decay 0.01, linear warmup 10% then linear decay),
micro-batches of `micro_batch` accumulated to an effective batch of
micro_batch x accum, inverse-frequency class weights in the cross-entropy, fixed seeds,
torch.set_num_threads(4). Micro-batches are built from length-sorted chunks of a seeded
shuffle, so a batch pads to similar lengths (CPU time is linear in padded tokens).

Output (all under data/models/verifier/, gitignored):
  model/       the checkpoint with the best validation macro-F1 so far (save_pretrained,
               + tokenizer + meta.json) — written atomically at the end of an epoch; the
               path app.verify.trained loads (and where a Kaggle-trained model goes)
  state.pt     full training state (model, optimiser, scheduler, RNG, position), saved
               every `save_every` optimiser steps and at every epoch end; a re-run with
               the same config resumes from it
  train.log    the log (also printed)
  splits.json  the claim ids of every split and the data report

Run (in the background; it takes hours on a laptop CPU):
    nohup uv run --locked python -m app.verify.train > data/models/verifier/nohup.out 2>&1 &
    make verify-train
    uv run python -m app.verify.train --time-steps 20   # timing/RAM probe, trains nothing to disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from app.verify import train_data as td

BASE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
OUT_DIR = Path("data/models/verifier")
MODEL_SUBDIR = "model"  # OUT_DIR / MODEL_SUBDIR is app.verify.trained.CHECKPOINT_DIR
LABEL_IDS = {lab: i for i, lab in enumerate(td.LABELS)}  # SUPPORT 0, CONTRADICT 1, NEI 2
SPLIT_KEYS = ("train_qids", "val_qids", "tuning_qids", "excluded_qids", "excluded_test_qids")


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = BASE_MODEL
    max_length: int = 512
    micro_batch: int = 2  # 2 x 8: same s/example as 4 x 4 on this CPU, peak RSS 4.35 vs 5.6 GB
    accum: int = 8
    lr: float = 2e-5
    epochs: int = 3
    warmup_frac: float = 0.1
    weight_decay: float = 0.01
    seed: int = 13
    threads: int = 4
    n_hard: int = 3  # v2 (Kaggle notebook v2): 3 hard negatives per claim, was 2
    random_frac: float = 0.25
    val_frac: float = 0.20  # v2: 20% of claims (grouped by doc) for validation, was 15%
    freeze_embeddings: bool = False
    freeze_layers: int = 0  # bottom encoder layers kept frozen (0 = train all)
    exclude_test_doc_overlap: bool = True  # drop train claims citing a test claim's doc
    save_every: int = 20  # optimiser steps between resumable state saves

    def signature(self) -> str:
        """What a resume must match: everything but the save cadence."""
        d = {k: v for k, v in asdict(self).items() if k != "save_every"}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


class Logger:
    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *parts) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(p) for p in parts)
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a") as fh:
                fh.write(line + "\n")


# --- data ------------------------------------------------------------------------------------


def load_train_splits(cfg: TrainConfig) -> dict:
    """Real SciFact train data -> train_data.build_splits (no Qdrant needed when the
    retrieval cache from verify_eval --tune exists; else SearchService runs once)."""
    from app.eval.rag_eval import sample_claims
    from app.eval.verify_eval import load_retrieval
    from app.ingest.corpus import load_claim_labels, load_documents, load_queries_qrels

    queries, qrels = load_queries_qrels(td.TRAIN_DATASET)
    labels = load_claim_labels(td.TRAIN_DATASET, query_ids=set(queries))
    tuning = td.tuning_ids(sample=sample_claims(queries, 100))
    top, _ = load_retrieval(td.TRAIN_DATASET, queries, sorted(queries))
    docs = {d["doc_id"]: d for d in load_documents()}
    # The test claims are read only to be kept out (see app.verify.train_data).
    t_queries, t_qrels = load_queries_qrels(td.TEST_DATASET)
    t_labels = load_claim_labels(td.TEST_DATASET, query_ids=set(t_queries))
    t_docs = set().union(*(td.cited_docs(q, t_labels, t_qrels) for q in t_queries))
    return td.build_splits(queries, labels, qrels, top, docs, tuning, val_frac=cfg.val_frac,
                           n_hard=cfg.n_hard, random_frac=cfg.random_frac, seed=cfg.seed,
                           test_texts=list(t_queries.values()),
                           test_docs=t_docs if cfg.exclude_test_doc_overlap else ())


def encode(tokenizer, pairs: Sequence[td.Pair], max_length: int) -> list[dict[str, list[int]]]:
    """{input_ids, token_type_ids} per pair: claim first, premise (title. abstract)
    truncated — the same encoding NLIVerifier._forward builds at inference (segment ids
    included: dropping them here would train on all-zero segments and infer on 0/1)."""
    from app.verify.nli import premise_text

    enc = tokenizer([p.claim for p in pairs], [premise_text(p.title, p.abstract) for p in pairs],
                    truncation="only_second", max_length=max_length)
    keys = [k for k in ("input_ids", "token_type_ids") if k in enc]
    return [{k: enc[k][i] for k in keys} for i in range(len(pairs))]


def epoch_batches(lengths: Sequence[int], micro_batch: int, chunk: int, seed: int, epoch: int) -> list[list[int]]:
    """Micro-batches of example indices for one epoch: a seeded shuffle, cut into chunks
    of `chunk`, each sorted by length and sliced into micro-batches. Deterministic in
    (seed, epoch), which is what lets a resumed run skip exactly the batches it did."""
    order = list(range(len(lengths)))
    random.Random(seed * 1000 + epoch).shuffle(order)
    batches = []
    for c in range(0, len(order), chunk):
        block = sorted(order[c : c + chunk], key=lambda i: lengths[i])
        batches.extend(block[b : b + micro_batch] for b in range(0, len(block), micro_batch))
    return batches


def collate(tokenizer, feats: Sequence[dict[str, list[int]]]):
    return tokenizer.pad([dict(f) for f in feats], return_tensors="pt")


# --- model -----------------------------------------------------------------------------------


def build_model(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=len(td.LABELS),
        id2label={i: lab for lab, i in LABEL_IDS.items()}, label2id=dict(LABEL_IDS),
        dtype=torch.float32,
    )
    base = getattr(model, model.base_model_prefix)
    if cfg.freeze_embeddings or cfg.freeze_layers:
        for p in base.embeddings.parameters():
            p.requires_grad = False
    for layer in base.encoder.layer[: cfg.freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    return tok, model


def evaluate(model, tokenizer, ids: Sequence[Sequence[int]], gold: Sequence[str], batch: int = 8) -> dict:
    """Pair-level scores on a held-out set: accuracy, macro-F1, per-label P/R/F1."""
    import torch

    from app.eval.verify_eval import class_scores

    model.eval()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]["input_ids"]))
    pred: list[str | None] = [None] * len(ids)
    with torch.inference_mode():
        for b in range(0, len(order), batch):
            idx = order[b : b + batch]
            logits = model(**collate(tokenizer, [ids[i] for i in idx])).logits
            for i, k in zip(idx, logits.argmax(-1).tolist(), strict=True):
                pred[i] = td.LABELS[k]
    model.train()
    return class_scores(list(gold), pred)


# --- training loop -----------------------------------------------------------------------------


def _save_state(path: Path, **state) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _save_best(out: Path, model, tokenizer, meta: dict) -> None:
    tmp = out / f"{MODEL_SUBDIR}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    best = out / MODEL_SUBDIR
    shutil.rmtree(best, ignore_errors=True)
    os.replace(tmp, best)


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def train(cfg: TrainConfig, out: Path = OUT_DIR, time_steps: int = 0, splits: dict | None = None) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    log = Logger(None if time_steps else out / "train.log")
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t_start = time.time()

    splits = splits or load_train_splits(cfg)
    rep = splits["report"]
    log(f"config {asdict(cfg)} sig={cfg.signature()}")
    log(f"data: {json.dumps(rep)}")
    if not time_steps:
        out.mkdir(parents=True, exist_ok=True)
        (out / "splits.json").write_text(json.dumps({
            k: splits[k] for k in SPLIT_KEYS if k in splits
        } | {"report": rep}, indent=1))

    tok, model = build_model(cfg)
    train_ids = encode(tok, splits["train"], cfg.max_length)
    val_ids = encode(tok, splits["val"], cfg.max_length)
    y = torch.tensor([LABEL_IDS[p.label] for p in splits["train"]])
    lengths = [len(x["input_ids"]) for x in train_ids]
    weights = torch.tensor([rep["class_weights"][lab] for lab in td.LABELS], dtype=torch.float32)
    log(f"tokens/pair: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}, "
        f"truncated {sum(n >= cfg.max_length for n in lengths)}/{len(lengths)}; "
        f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    chunk = cfg.micro_batch * cfg.accum * 4
    n_micro = math.ceil(len(train_ids) / cfg.micro_batch)
    steps_per_epoch = math.ceil(n_micro / cfg.accum)
    total_steps = steps_per_epoch * cfg.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not (n.endswith("bias") or "LayerNorm" in n)]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith("bias") or "LayerNorm" in n)]
    assert len(decay) + len(no_decay) == len(params)
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_frac * total_steps), total_steps)

    start_epoch, start_micro, best, history = 0, 0, None, []
    state_path = out / "state.pt"
    if not time_steps and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st.get("signature") != cfg.signature():
            raise SystemExit(f"{state_path} was saved under a different config; move it away to start over")
        model.load_state_dict(st["model"])
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        start_epoch, start_micro = st["epoch"], st["micro"]
        best, history = st.get("best"), st.get("history", [])
        log(f"RESUMED from {state_path}: epoch {start_epoch + 1}, micro-batch {start_micro}/{n_micro}, best {best}")
        del st

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    model.train()
    done_micro, t_loop, peak = 0, time.time(), 0.0
    for epoch in range(start_epoch, cfg.epochs):
        batches = epoch_batches(lengths, cfg.micro_batch, chunk, cfg.seed, epoch)
        running, seen = 0.0, 0
        first = start_micro if epoch == start_epoch else 0
        for m in range(first, len(batches)):
            idx = batches[m]
            enc = collate(tok, [train_ids[i] for i in idx])
            logits = model(**enc).logits
            loss = loss_fn(logits, y[idx])
            (loss / cfg.accum).backward()
            running += loss.item() * len(idx)
            seen += len(idx)
            done_micro += 1
            peak = max(peak, rss_mb())
            last_in_epoch = m == len(batches) - 1
            if (m + 1) % cfg.accum == 0 or last_in_epoch:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step = epoch * steps_per_epoch + (m // cfg.accum) + 1
                if not time_steps and step % cfg.save_every == 0 and not last_in_epoch:
                    _save_state(state_path, model=model.state_dict(), optim=optim.state_dict(),
                                sched=sched.state_dict(), torch_rng=torch.get_rng_state(),
                                epoch=epoch, micro=m + 1, best=best, history=history,
                                signature=cfg.signature())
            if done_micro % 50 == 0 or (time_steps and done_micro == time_steps):
                el = time.time() - t_loop
                per = el / done_micro
                left_micro = (cfg.epochs - epoch - 1) * len(batches) + (len(batches) - m - 1)
                eta = per * left_micro + (cfg.epochs - epoch) * per * len(val_ids) / cfg.micro_batch * 0.35
                log(f"epoch {epoch + 1} micro {m + 1}/{len(batches)} loss {running / max(seen, 1):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} | {per / cfg.micro_batch:.2f} s/example, "
                    f"ETA {eta / 3600:.2f} h, rss {rss_mb():.0f} MB (peak {peak:.0f})")
            if time_steps and done_micro >= time_steps:
                el = time.time() - t_loop
                return {"s_per_example": el / (done_micro * cfg.micro_batch), "peak_rss_mb": peak,
                        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        "est_hours": el / done_micro * len(batches) * cfg.epochs / 3600}
        t_val = time.time()
        vs = evaluate(model, tok, val_ids, [p.label for p in splits["val"]])
        entry = {"epoch": epoch + 1, "train_loss": round(running / max(seen, 1), 5),
                 "val_accuracy": vs["accuracy"], "val_macro_f1": vs["macro_f1"],
                 "val_per_label": vs["per_label"], "val_s": round(time.time() - t_val, 1),
                 "elapsed_h": round((time.time() - t_start) / 3600, 3)}
        history.append(entry)
        log(f"epoch {epoch + 1} done: {json.dumps(entry)}")
        if best is None or vs["macro_f1"] > best["val_macro_f1"]:
            best = {"epoch": epoch + 1, "val_macro_f1": vs["macro_f1"], "val_accuracy": vs["accuracy"]}
            _save_best(out, model, tok, {
                "base_model": cfg.base_model, "max_length": cfg.max_length, "pair_order": "claim_first",
                "labels": list(td.LABELS), "config": asdict(cfg), "config_signature": cfg.signature(),
                "best": best, "history": history, "data": rep, "val_qids": splits["val_qids"],
            })
            log(f"  new best -> {out / MODEL_SUBDIR}")
        _save_state(state_path, model=model.state_dict(), optim=optim.state_dict(),
                    sched=sched.state_dict(), torch_rng=torch.get_rng_state(),
                    epoch=epoch + 1, micro=0, best=best, history=history, signature=cfg.signature())
    summary = {"best": best, "history": history, "hours": round((time.time() - t_start) / 3600, 3),
               "peak_rss_mb": round(max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))}
    if best is not None and (out / MODEL_SUBDIR).exists():
        meta = json.loads((out / MODEL_SUBDIR / "meta.json").read_text())
        meta["final_history"] = history
        meta["train_hours"] = summary["hours"]
        (out / MODEL_SUBDIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"finished: {json.dumps(summary)}")
    return summary


def _parse_args(argv: Sequence[str]) -> tuple[TrainConfig, int]:
    ap = argparse.ArgumentParser(prog="python -m app.verify.train", description=__doc__.split("\n")[0])
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=getattr(defaults, f.name))
        else:
            kind = {"int": int, "float": float, "str": str}.get(str(f.type), str)
            ap.add_argument(flag, type=kind, default=getattr(defaults, f.name))
    ap.add_argument("--time-steps", type=int, default=0, help="timing/RAM probe: run N micro-batches, save nothing")
    a = vars(ap.parse_args(list(argv)))
    time_steps = a.pop("time_steps")
    return TrainConfig(**a), time_steps


def _oom_first() -> None:
    """Volunteer this process as the OOM killer's first choice (Linux; raising one's own
    score needs no privilege), so a memory squeeze ends the training run, which resumes
    from state.pt, rather than whatever else the laptop is running."""
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass


def main(argv: Sequence[str] = ()) -> None:
    cfg, time_steps = _parse_args(argv)
    _oom_first()
    res = train(cfg, time_steps=time_steps)
    if time_steps:
        print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1:])
