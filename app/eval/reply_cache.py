"""On-disk cache of single LLM replies, keyed by the exact request.

Shared by app.eval.rag_secondlook (its post-hoc re-ask / second-look experiments) and
app.eval.rag_eval (the re-ask post-step of the canonical eval): both key a re-ask reply
as ``reask|<query id>|<sha16 of [model, max_tokens, messages]>``, so a reply fetched by
either is a cache hit for the other and is never paid for (or quota-spent) twice.
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

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, record: dict) -> None:
        self.entries[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": self.entries}, indent=1))
        os.replace(tmp, self.path)
