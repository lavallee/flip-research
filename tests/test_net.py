"""net.py against a throwaway local HTTP server (no outside network)."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flip_research import net
from flip_research.errors import ResearchError

ROUTES = {
    "/big": (200, "text/plain", b"x" * 5000),
    "/latin": (200, "text/html; charset=iso-8859-1", "caf\xe9".encode("latin-1")),
    "/missing": (404, "text/plain", b"nope"),
    "/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /private\n"),
    "/ua": (200, "text/plain", b""),
}
SEEN = {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        SEEN["ua"] = self.headers.get("User-Agent")
        status, ctype, body = ROUTES.get(self.path, (404, "text/plain", b"nope"))
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_size_cap_and_user_agent(server):
    r = net.http_get(server + "/big", max_bytes=1000)
    assert len(r.body) == 1000 and r.truncated
    assert SEEN["ua"] == net.USER_AGENT
    assert net.USER_AGENT.startswith("flip-research/")
    assert "@" not in net.USER_AGENT and "http" not in net.USER_AGENT


def test_http_error_is_returned_not_raised(server):
    assert net.http_get(server + "/missing").status == 404


def test_charset_decoding(server):
    r = net.http_get(server + "/latin")
    assert r.charset == "iso-8859-1" and net.decode(r) == "café"


def test_connection_refused_is_a_network_error():
    with pytest.raises(ResearchError) as e:
        net.http_get("http://127.0.0.1:1/", timeout=2)
    assert e.value.kind == "network"


def test_public_guard_refuses_local_addresses(server):
    with pytest.raises(ResearchError) as e:
        net.http_get(server + "/big", guarded=True)
    assert e.value.kind == "refused_url"
    with pytest.raises(ResearchError):
        net.check_public("file:///etc/passwd")


def test_robots_rules(server):
    robots = net.Robots(fetch=lambda url, **k: net.http_get(url, max_bytes=k.get("max_bytes", 1)))
    assert robots.allowed(server + "/public")
    assert not robots.allowed(server + "/private/page")


def test_robots_statuses():
    def serving(status, body=b""):
        return lambda url, **k: net.Response(status=status, url=url, body=body)

    assert net.Robots(serving(404)).allowed("https://a.example/x")  # no file: allowed
    assert net.Robots(serving(403)).allowed("https://a.example/x")  # RFC 9309: 4xx allowed
    assert not net.Robots(serving(503)).allowed("https://a.example/x")  # 5xx: stay away

    def down(url, **k):
        raise ResearchError("network", "down")

    assert not net.Robots(down).allowed("https://a.example/x")


def test_robots_cached_per_origin():
    calls = []

    def fetch(url, **k):
        calls.append(url)
        return net.Response(status=200, url=url, body=b"User-agent: *\nDisallow: /no\n")

    r = net.Robots(fetch)
    r.allowed("https://a.example/1")
    r.allowed("https://a.example/2")
    r.allowed("https://b.example/1")
    assert calls == ["https://a.example/robots.txt", "https://b.example/robots.txt"]
