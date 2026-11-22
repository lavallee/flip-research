"""Search providers. Each is a small function `(query, k, env) -> list[dict]` returning
raw hits with url/title/snippet; `find` normalises, de-duplicates and ranks them.

Tests replace entries in PROVIDERS. Nothing here invents a result: a provider that
cannot answer raises ResearchError and the command exits non-zero.
"""

from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from . import net
from .errors import ResearchError
from .text import norm_ws

AUTO_ORDER = ("searxng", "brave", "tavily", "duckduckgo")
_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid", "yclid"}


def normalize_url(url: str) -> str:
    """A key for de-duplication: lowercase host, no fragment, no default port,
    no tracking parameters, no trailing slash."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING]
    path = parts.path.rstrip("/") or ""
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))


# --- DuckDuckGo HTML (keyless, best effort) -------------------------------------------

class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict] = []
        self._mode: str | None = None  # "title" | "snippet"
        self._tag = ""
        self._depth = 0
        self._buf: list[str] = []
        self._cur: dict | None = None
        self.saw_results_container = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if "result" in classes or "results" in classes or "serp__results" in classes:
            self.saw_results_container = True
        if self._mode:
            if tag == self._tag:
                self._depth += 1
            return
        if tag == "a" and "result__a" in classes:
            self._cur = {"url": _unwrap(a.get("href") or ""), "title": "", "snippet": ""}
            self._begin("title", tag)
        elif "result__snippet" in classes and self._cur is not None:
            self._begin("snippet", tag)

    def _begin(self, mode: str, tag: str) -> None:
        self._mode, self._tag, self._depth, self._buf = mode, tag, 1, []

    def handle_endtag(self, tag):
        if not self._mode or tag != self._tag:
            return
        self._depth -= 1
        if self._depth:
            return
        text = norm_ws("".join(self._buf))
        if self._mode == "title" and self._cur is not None:
            self._cur["title"] = text
            self.hits.append(self._cur)
        elif self._mode == "snippet" and self._cur is not None:
            self._cur["snippet"] = text
        self._mode = None

    def handle_data(self, data):
        if self._mode:
            self._buf.append(data)


def _unwrap(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect (`/l/?uddg=<url>`); unwrap it."""
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if parts.path == "/l/" or parts.path.endswith("/l/"):
        for k, v in parse_qsl(parts.query):
            if k == "uddg":
                return unquote(v)
    return href


def parse_duckduckgo(html: str) -> list[dict]:
    """Parse DuckDuckGo's HTML results page. Raises ResearchError if the page is a
    challenge, or has a shape this parser does not recognise."""
    low = html.lower()
    if "anomaly" in low or "unusual traffic" in low or "captcha" in low:
        raise ResearchError("blocked", "DuckDuckGo served a bot challenge instead of results")
    parser = _DDGParser()
    parser.feed(html)
    parser.close()
    hits = [h for h in parser.hits if urlsplit(h["url"]).hostname
            and not (urlsplit(h["url"]).hostname or "").endswith("duckduckgo.com")]
    if not hits and "no results" not in low and "no more results" not in low:
        raise ResearchError(
            "unexpected_page",
            "DuckDuckGo returned a page with no recognisable results; its markup may have changed",
        )
    return hits


def search_duckduckgo(query: str, k: int, env) -> list[dict]:
    resp = net.http_get("https://html.duckduckgo.com/html/?q=" + quote(query),
                        headers={"Accept": "text/html"})
    if resp.status in (202, 403, 418, 429):
        raise ResearchError(
            "blocked",
            f"DuckDuckGo refused the request (HTTP {resp.status}); the keyless provider is "
            "best effort. Set FLIP_RESEARCH_SEARXNG_URL, BRAVE_API_KEY or TAVILY_API_KEY.",
        )
    if resp.status != 200:
        raise ResearchError("unavailable", f"DuckDuckGo answered HTTP {resp.status}")
    return parse_duckduckgo(net.decode(resp))


# --- JSON providers -------------------------------------------------------------------

def _json(resp: net.Response, who: str):
    if resp.status in (401, 403, 429):
        kind = "blocked" if resp.status == 403 else "auth" if resp.status == 401 else "rate_limited"
        raise ResearchError(kind, f"{who} answered HTTP {resp.status}")
    if resp.status != 200:
        raise ResearchError("unavailable", f"{who} answered HTTP {resp.status}")
    try:
        return json.loads(net.decode(resp))
    except ValueError as exc:
        raise ResearchError("bad_response", f"{who} did not return JSON") from exc


def _items(data, key: str, who: str) -> list[dict]:
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, list):
        raise ResearchError("bad_response", f"{who} response has no '{key}' list")
    return [x for x in value if isinstance(x, dict)]


def search_searxng(query: str, k: int, env) -> list[dict]:
    base = env["FLIP_RESEARCH_SEARXNG_URL"].rstrip("/")
    resp = net.http_get(f"{base}/search?" + urlencode({"q": query, "format": "json"}),
                        headers={"Accept": "application/json"})
    if resp.status == 403:
        raise ResearchError(
            "blocked", "the SearXNG instance refused the JSON API (HTTP 403); enable "
            "`json` under search.formats in its settings.yml")
    data = _json(resp, "SearXNG")
    return [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")}
            for r in _items(data, "results", "SearXNG")]


def search_brave(query: str, k: int, env) -> list[dict]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urlencode(
        {"q": query, "count": min(max(k, 1), 20)})
    resp = net.http_get(url, headers={"Accept": "application/json",
                                      "X-Subscription-Token": env["BRAVE_API_KEY"]})
    data = _json(resp, "Brave Search")
    web = data.get("web") if isinstance(data, dict) else None
    if not isinstance(web, dict):
        return []  # Brave omits "web" when there are no web results
    return [{"url": r.get("url", ""), "title": r.get("title", ""),
             "snippet": r.get("description", "")}
            for r in _items(web, "results", "Brave Search")]


def search_tavily(query: str, k: int, env) -> list[dict]:
    body = json.dumps({"query": query, "max_results": min(max(k, 1), 20)}).encode()
    resp = net.http_get("https://api.tavily.com/search", data=body, headers={
        "Content-Type": "application/json", "Accept": "application/json",
        "Authorization": f"Bearer {env['TAVILY_API_KEY']}"})
    data = _json(resp, "Tavily")
    return [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")}
            for r in _items(data, "results", "Tavily")]


PROVIDERS = {
    "searxng": search_searxng,
    "brave": search_brave,
    "tavily": search_tavily,
    "duckduckgo": search_duckduckgo,
}

_NEEDS = {
    "searxng": "FLIP_RESEARCH_SEARXNG_URL",
    "brave": "BRAVE_API_KEY",
    "tavily": "TAVILY_API_KEY",
}


def load_fixture(path: str) -> dict:
    """Test hook: `FLIP_RESEARCH_PROVIDER=fixture:<path>` reads results (and optionally
    pages for `ask`, or an injected error) from a JSON file. No network is used."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResearchError("config", f"cannot read fixture provider file {path}: {exc}") from exc
    if isinstance(data, list):
        data = {"results": data}
    if not isinstance(data, dict):
        raise ResearchError("config", "fixture provider file must hold an object or a list")
    return data


def provider_status(env) -> list[dict]:
    """For `doctor`: every real provider, whether it is configured, and why."""
    rows = []
    for name in AUTO_ORDER:
        var = _NEEDS.get(name)
        if var:
            ok = bool(env.get(var))
            reason = f"{var} is set" if ok else f"{var} is not set"
        else:
            ok = True
            reason = "keyless, best effort; may be refused"
        rows.append({"name": name, "configured": ok, "reason": reason})
    return rows


def choose(name: str | None, env) -> str:
    """Resolve the provider name: --provider, then FLIP_RESEARCH_PROVIDER, then the
    first configured provider in AUTO_ORDER. No silent fallback after that."""
    name = name or env.get("FLIP_RESEARCH_PROVIDER") or None
    if name:
        if name.startswith("fixture:"):
            return name
        if name not in PROVIDERS:
            raise ResearchError(
                "config", f"unknown provider '{name}'; choose one of {', '.join(AUTO_ORDER)}")
        var = _NEEDS.get(name)
        if var and not env.get(var):
            raise ResearchError("config", f"provider '{name}' needs {var} to be set")
        return name
    for cand in AUTO_ORDER:
        var = _NEEDS.get(cand)
        if var is None or env.get(var):
            return cand
    raise ResearchError("config", "no provider available")  # unreachable: duckduckgo is keyless


def find(query: str, k: int = 10, provider: str | None = None, env=None) -> dict:
    env = os.environ if env is None else env
    if not query.strip():
        raise ResearchError("usage", "the query is empty")
    if k < 1:
        raise ResearchError("usage", "--k must be at least 1")
    name = choose(provider, env)
    did = [f"searched {name} for {query!r}"]
    if name.startswith("fixture:"):
        data = load_fixture(name[len("fixture:"):])
        if isinstance(data.get("error"), dict):
            err = data["error"]
            raise ResearchError(str(err.get("kind", "unavailable")), str(err.get("message", "")))
        raw = [r for r in data.get("results", []) if isinstance(r, dict)]
    else:
        raw = PROVIDERS[name](query, k, env)
    results, seen, dropped = [], set(), 0
    for r in raw:
        url = str(r.get("url") or "").strip()
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).hostname:
            dropped += 1
            continue
        key = normalize_url(url)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        results.append({"url": url, "title": norm_ws(str(r.get("title") or "")),
                        "snippet": norm_ws(str(r.get("snippet") or ""))})
    if dropped:
        did.append(f"dropped {dropped} duplicate or non-http(s) result(s)")
    results = results[:k]
    for i, r in enumerate(results, 1):
        r["provider"] = name.split(":")[0]
        r["rank"] = i
    did.append(f"returned {len(results)} result(s)")
    return {"results": results, "query": query, "provider": name.split(":")[0], "did": did}


