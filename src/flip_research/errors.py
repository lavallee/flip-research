"""One error type: a kind a script can branch on and a sentence a person can read."""

from __future__ import annotations


class ResearchError(Exception):
    """Raised for every expected failure. `kind` is stable; `message` is for humans."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    def as_json(self) -> dict:
        return {"error": self.message, "error_kind": self.kind}
