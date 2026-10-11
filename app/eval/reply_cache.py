"""On-disk cache of single LLM replies, keyed by the exact request.

Used by app.eval.rag_eval's re-ask post-step. Its file
(``data/eval_cache/secondlook/<dataset>.json``) was first written by the post-hoc re-ask
experiment (app.eval.rag_secondlook, archived in the experiments-archive tag); the path is
kept so one file holds every re-ask reply fetched for a dataset.

Key: ``<kind>|v2|<query id>|<sha16 of [request, max_tokens, messages]>``
(``ReplyCache.key_v2``), where ``request`` is a mapping that fingerprints the endpoint and
sampling policy (for the re-ask: rag_eval.reask_request_fingerprint — provider, base URL,
model, resolved reasoning effort, temperature, routing), so the same messages sent to
another provider, base URL or model are a miss, never a silent cross-provider hit.

The legacy v1 key (model id + budget + messages, nothing about who served it) is no longer
read or written: every v1 entry came from the OpenRouter Ling generator, which was
withdrawn upstream, so no current endpoint could reuse them. Entries already in the file
are left as they are; nothing is rewritten in place.
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
