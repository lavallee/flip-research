"""Stdlib HTTP: one GET/POST helper, a size cap, a public-host guard, robots.txt.

The User-Agent is a plain tool identifier. It carries no contact details and
does not pretend to be a browser.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from . import __version__
from .errors import ResearchError

USER_AGENT = f"flip-research/{__version__}"
TIMEOUT = 15.0
MAX_BYTES = 1_000_000


@dataclass
class Response:
    status: int
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    truncated: bool = False

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def charset(self) -> str | None:
        for part in self.headers.get("content-type", "").split(";")[1:]:
            k, _, v = part.strip().partition("=")
            if k.lower() == "charset" and v:
                return v.strip("\"' ")
        return None


def check_public(url: str) -> None:
    """Refuse to fetch a page from a loopback, private or link-local address.

    Search results are untrusted; a result must not be able to aim this tool at
    the user's own network. (Search *providers* are exempt: a local SearXNG is
    legitimate.)
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ResearchError("refused_url", f"not an http(s) URL: {url}")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ResearchError("network", f"cannot resolve {parts.hostname}: {exc}") from exc
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global:
            raise ResearchError("refused_url", f"{parts.hostname} is not a public address")


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request: urllib.request.Request, timeout: float, guarded: bool):
    if guarded:
        opener = urllib.request.build_opener(_GuardedRedirect)
        return opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float = TIMEOUT,
    max_bytes: int = MAX_BYTES,
    guarded: bool = False,
) -> Response:
    """GET (or POST when `data` is given). HTTP error statuses are returned, not raised.

    Only transport failures raise ResearchError(kind="network").
    """
    if guarded:
        check_public(url)
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with _open(req, timeout, guarded) as resp:
            body = resp.read(max_bytes + 1)
            return _response(resp.status, resp.geturl(), resp.headers, body, max_bytes)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(max_bytes + 1)
        except OSError:
            body = b""
        return _response(exc.code, url, exc.headers, body, max_bytes)
    except ResearchError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ResearchError("network", f"request to {urlsplit(url).netloc} failed: {reason}") from exc


def _response(status: int, url: str, headers, body: bytes, max_bytes: int) -> Response:
    truncated = len(body) > max_bytes
    return Response(
        status=status,
        url=url,
        headers={k.lower(): v for k, v in headers.items()},
        body=body[:max_bytes],
        truncated=truncated,
    )


def decode(resp: Response) -> str:
    for enc in (resp.charset, "utf-8"):
        if not enc:
            continue
        try:
            return resp.body.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return resp.body.decode("utf-8", errors="replace")


class Robots:
    """robots.txt for this tool's User-Agent, cached per origin for one run.

    Follows RFC 9309: a missing file or any 4xx means allowed; a 5xx or an
    unreachable server for the robots file itself is treated as "do not fetch".
    """

    def __init__(self, fetch=http_get):
        self._fetch = fetch
        self._cache: dict[str, RobotFileParser | bool] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        rules = self._cache[origin]
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(USER_AGENT, url)

    def _load(self, origin: str) -> RobotFileParser | bool:
        try:
            resp = self._fetch(f"{origin}/robots.txt", guarded=True, max_bytes=500_000)
        except ResearchError:
            return False
        if resp.status >= 500:
            return False
        if resp.status >= 400:
            return True
        parser = RobotFileParser()
        parser.parse(decode(resp).splitlines())
        return parser
