"""On-disk cache of single LLM replies, keyed by the exact request.

Used by app.eval.rag_eval's re-ask post-step. Its file
(``data/eval_cache/secondlook/<dataset>.json``) was first written by the post-hoc re-ask
experiment (app.eval.rag_secondlook, archived in the experiments-archive tag), so the
replies that experiment fetched — the ones behind the committed eval/results/rag.json —
are cache hits for the eval and are never fetched (or quota-spent) twice.

Two key formats:

* **v1 (legacy)** — ``<kind>|<query id>|<sha16 of [model, max_tokens, messages]>``
  (``ReplyCache.key``). It names the model id but not who served it: the same model id
  and messages sent to another provider / base URL, with another reasoning effort,
  temperature policy or routing, would collide. Every re-ask reply cached before the v2
  format (the 15 behind the committed eval/results/rag.json among them) is under v1. The
  v1 key code must stay: removing it turns those replies into cache misses, and the
  canonical run could no longer be reproduced with zero LLM calls.
* **v2** — ``<kind>|v2|<query id>|<sha16 of [request, max_tokens, messages]>``
  (``ReplyCache.key_v2``), where ``request`` is a mapping that fingerprints the endpoint
  and sampling policy (for the re-ask: rag_eval.reask_request_fingerprint — provider,
  base URL, model, resolved reasoning effort, temperature, routing). New re-ask replies
  are written only under v2.

Migration: a re-ask lookup tries the v2 key first and falls back to the v1 key ONLY for
the one endpoint every v1 re-ask entry was fetched with — the canonical OpenRouter Ling
endpoint (rag_eval.is_legacy_reask_endpoint). Any other endpoint never reads a v1 entry,
so a Groq (or paid, or differently-routed) run with the same model id + messages is a
miss, not a silent cross-provider hit. Entries are never rewritten in place.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class ReplyCache:
    """Replies keyed by the exact request; saved atomically after every new entry."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            blob = json.loads(self.path.read_text())
            self.entries: dict[str, dict] = blob["entries"] if isinstance(blob.get("entries"), dict) else {}
        except (OSError, ValueError, AttributeError):
            self.entries = {}

    @staticmethod
    def key(kind: str, qid: str, model: str, max_tokens: int, messages: Sequence[Mapping]) -> str:
        return f"{kind}|{qid}|{_sha([model, max_tokens, list(messages)])[:16]}"

    @staticmethod
    def key_v2(
        kind: str, qid: str, request: Mapping, max_tokens: int, messages: Sequence[Mapping]
    ) -> str:
        """The v2 key: ``request`` fingerprints everything about the endpoint that can
        change the reply (see the module docstring); it must be JSON-serialisable."""
        return f"{kind}|v2|{qid}|{_sha([dict(request), max_tokens, list(messages)])[:16]}"

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, record: dict) -> None:
        self.entries[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": self.entries}, indent=1))
        os.replace(tmp, self.path)
