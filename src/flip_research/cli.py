"""Command line: one JSON document on stdout, or one JSON error line on stderr."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, ask, model, providers
from .errors import ResearchError


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # usage errors are JSON too
        print(json.dumps(ResearchError("usage", message).as_json()), file=sys.stderr)
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="flip-research", description=__doc__)
    p.add_argument("--version", action="version", version=f"flip-research {__version__}")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)

    f = sub.add_parser("find", help="candidate sources for a query")
    f.add_argument("query")
    f.add_argument("--k", type=int, default=10, help="results to return (default 10)")
    f.add_argument("--provider", help="searxng, brave, tavily or duckduckgo")

    a = sub.add_parser("ask", help="a cited answer built from fetched pages")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=5, help="pages to fetch and read (default 5)")
    a.add_argument("--provider")

    v = sub.add_parser("verify", help="a verdict on a claim, with quotes")
    v.add_argument("claim")
    v.add_argument("--k", type=int, default=5)
    v.add_argument("--provider")

    sub.add_parser("doctor", help="what would be used, without a paid call")
    return p


def doctor(env=None) -> dict:
    env = os.environ if env is None else env
    rows = providers.provider_status(env)
    try:
        chosen = providers.choose(None, env)
        pr = {"selected": chosen, "reason": _why(chosen, env)}
    except ResearchError as exc:
        pr = {"selected": None, "error_kind": exc.kind, "reason": exc.message}
    pr["available"] = rows
    return {"version": __version__, "provider": pr, "model": model.describe(env),
            "note": "doctor makes no network call and no paid call"}


def _why(name: str, env) -> str:
    if name.startswith("fixture:"):
        return "FLIP_RESEARCH_PROVIDER selects a fixture file (test hook)"
    if env.get("FLIP_RESEARCH_PROVIDER"):
        return f"FLIP_RESEARCH_PROVIDER={name}"
    return next(r["reason"] for r in providers.provider_status(env) if r["name"] == name) + \
        "; first configured provider in the default order"


def run(args: argparse.Namespace) -> dict:
    if args.command == "doctor":
        return doctor()
    if getattr(args, "k", 1) < 1:
        raise ResearchError("usage", "--k must be at least 1")
    if args.command == "find":
        return providers.find(args.query, args.k, args.provider)
    if args.command == "ask":
        return ask.ask(args.question, args.k, args.provider)
    return ask.verify(args.claim, args.k, args.provider)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except ResearchError as exc:
        print(json.dumps(exc.as_json(), ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001  one JSON line, never a traceback
        err = ResearchError("internal", f"{type(exc).__name__}: {exc}")
        print(json.dumps(err.as_json(), ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0
