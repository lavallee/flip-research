import json

import pytest

from flip_research import net, providers
from flip_research.errors import ResearchError


def resp(status=200, body="", headers=None):
    data = body if isinstance(body, bytes) else (
        json.dumps(body) if not isinstance(body, str) else body).encode()
    return net.Response(status=status, url="https://x", headers=headers or {}, body=data)


# --- DuckDuckGo parser ----------------------------------------------------------------

def test_duckduckgo_parser_unwraps_redirects_and_skips_ads(fixture_text):
    hits = providers.parse_duckduckgo(fixture_text("ddg_results.html"))
    assert [h["url"] for h in hits] == [
        "https://example.org/okf/spec?utm_source=news",
        "https://example.org/okf/spec/#intro",
        "https://other.example.com/format",
    ]
    assert hits[0]["title"] == "Open Knowledge Format — Specification"
    assert hits[0]["snippet"] == (
        "The open knowledge format defines a bundle of markdown pages & manifests.")
    assert hits[2]["snippet"] == "A second source with no anchor snippet."


def test_duckduckgo_challenge_is_blocked(fixture_text):
    with pytest.raises(ResearchError) as e:
        providers.parse_duckduckgo(fixture_text("ddg_challenge.html"))
    assert e.value.kind == "blocked"


def test_duckduckgo_empty_page_is_empty_not_error(fixture_text):
    assert providers.parse_duckduckgo(fixture_text("ddg_empty.html")) == []


def test_duckduckgo_unrecognised_page_is_an_error():
    with pytest.raises(ResearchError) as e:
        providers.parse_duckduckgo("<html><body><p>Something else entirely</p></body></html>")
    assert e.value.kind == "unexpected_page"


@pytest.mark.parametrize("status", [202, 403, 429])
def test_duckduckgo_refusal_statuses_are_blocked(monkeypatch, status):
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(status, "nope"))
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == "blocked"
    assert "FLIP_RESEARCH_SEARXNG_URL" in e.value.message


def test_duckduckgo_server_error_is_unavailable(monkeypatch):
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(503, "x"))
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == "unavailable"


def test_duckduckgo_find_end_to_end(monkeypatch, fixture_text):
    monkeypatch.setattr(net, "http_get",
                        lambda *a, **k: resp(200, fixture_text("ddg_results.html")))
    out = providers.find("open knowledge format")
    assert out["provider"] == "duckduckgo"
    # the second URL is the same page as the first once normalised
    assert [r["rank"] for r in out["results"]] == [1, 2]
    assert out["results"][0]["provider"] == "duckduckgo"
    assert any("duplicate" in d for d in out["did"])


# --- JSON providers -------------------------------------------------------------------

def test_searxng(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen["url"] = url
        return resp(200, {"results": [{"url": "https://a.example/x", "title": "A",
                                       "content": "snippet a"}]})

    monkeypatch.setattr(net, "http_get", fake)
    monkeypatch.setenv("FLIP_RESEARCH_SEARXNG_URL", "http://localhost:8080/")
    out = providers.find("hello world")
    assert seen["url"] == "http://localhost:8080/search?q=hello+world&format=json"
    assert out["provider"] == "searxng"
    assert out["results"][0]["snippet"] == "snippet a"


def test_searxng_json_disabled_is_blocked(monkeypatch):
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(403, "forbidden"))
    monkeypatch.setenv("FLIP_RESEARCH_SEARXNG_URL", "http://localhost:8080")
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == "blocked"
    assert "formats" in e.value.message


def test_brave(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(url=url, headers=kw["headers"])
        return resp(200, {"web": {"results": [
            {"url": "https://b.example", "title": "B", "description": "desc b"}]}})

    monkeypatch.setattr(net, "http_get", fake)
    monkeypatch.setenv("BRAVE_API_KEY", "k-brave")
    out = providers.find("q", k=3)
    assert seen["headers"]["X-Subscription-Token"] == "k-brave"
    assert "count=3" in seen["url"]
    assert out["results"][0]["snippet"] == "desc b"


def test_brave_no_web_results_is_empty(monkeypatch):
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(200, {"query": {}}))
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    assert providers.find("q")["results"] == []


def test_tavily(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(kw)
        return resp(200, {"results": [{"url": "https://t.example", "title": "T",
                                       "content": "body t"}]})

    monkeypatch.setattr(net, "http_get", fake)
    monkeypatch.setenv("TAVILY_API_KEY", "k-tav")
    out = providers.find("q")
    assert seen["headers"]["Authorization"] == "Bearer k-tav"
    assert json.loads(seen["data"])["query"] == "q"
    assert out["provider"] == "tavily"


@pytest.mark.parametrize("status,kind", [(401, "auth"), (429, "rate_limited"),
                                         (500, "unavailable")])
def test_json_provider_http_errors(monkeypatch, status, kind):
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(status, "x"))
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == kind


def test_json_provider_bad_json_and_bad_shape(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(200, "<html>"))
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == "bad_response"
    monkeypatch.setattr(net, "http_get", lambda *a, **k: resp(200, {"nope": 1}))
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == "bad_response"


def test_network_failure_propagates(monkeypatch):
    def boom(*a, **k):
        raise ResearchError("network", "request failed")

    monkeypatch.setattr(net, "http_get", boom)
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    with pytest.raises(ResearchError) as e:
        providers.find("q")
    assert e.value.kind == "network"


# --- selection, normalisation, validation ---------------------------------------------

def test_default_order(monkeypatch):
    assert providers.choose(None, {}) == "duckduckgo"
    assert providers.choose(None, {"TAVILY_API_KEY": "x"}) == "tavily"
    assert providers.choose(None, {"TAVILY_API_KEY": "x", "BRAVE_API_KEY": "y"}) == "brave"
    env = {"TAVILY_API_KEY": "x", "BRAVE_API_KEY": "y", "FLIP_RESEARCH_SEARXNG_URL": "http://s"}
    assert providers.choose(None, env) == "searxng"


def test_explicit_choice_beats_order_and_env_var():
    env = {"BRAVE_API_KEY": "y", "FLIP_RESEARCH_PROVIDER": "duckduckgo"}
    assert providers.choose(None, env) == "duckduckgo"
    assert providers.choose("brave", env) == "brave"


def test_explicit_provider_without_key_is_a_config_error():
    with pytest.raises(ResearchError) as e:
        providers.choose("brave", {})
    assert e.value.kind == "config" and "BRAVE_API_KEY" in e.value.message


def test_unknown_provider():
    with pytest.raises(ResearchError) as e:
        providers.choose("bing", {})
    assert e.value.kind == "config"


def test_empty_query_and_bad_k():
    for args in [("  ", 5), ("q", 0)]:
        with pytest.raises(ResearchError) as e:
            providers.find(*args, provider="duckduckgo")
        assert e.value.kind == "usage"


def test_normalize_url():
    n = providers.normalize_url
    assert n("HTTPS://Example.org:443/a/b/?utm_source=x&id=1#frag") == "https://example.org/a/b?id=1"
    assert n("http://example.org:8080/") == "http://example.org:8080"
    assert n("https://example.org/a") == n("https://example.org/a/")


def test_find_dedupes_filters_and_ranks(write_json):
    path = write_json({"results": [
        {"url": "https://a.example/p", "title": " A ", "snippet": "x"},
        {"url": "https://A.example/p/?utm_campaign=z", "title": "dup"},
        {"url": "ftp://files.example/x", "title": "ftp"},
        {"url": "", "title": "no url"},
        {"url": "https://b.example", "title": "B"},
        {"url": "https://c.example", "title": "C"},
    ]})
    out = providers.find("q", k=2, provider=f"fixture:{path}")
    assert [(r["url"], r["rank"]) for r in out["results"]] == [
        ("https://a.example/p", 1), ("https://b.example", 2)]
    assert out["results"][0]["title"] == "A"
    assert out["provider"] == "fixture"
    assert set(out) == {"results", "query", "provider", "did"}
    assert set(out["results"][0]) == {"url", "title", "snippet", "provider", "rank"}


def test_fixture_errors(tmp_path, write_json):
    with pytest.raises(ResearchError) as e:
        providers.find("q", provider=f"fixture:{tmp_path / 'missing.json'}")
    assert e.value.kind == "config"
    bad = write_json("just a string")
    with pytest.raises(ResearchError) as e:
        providers.find("q", provider=f"fixture:{bad}")
    assert e.value.kind == "config"
    inj = write_json({"error": {"kind": "blocked", "message": "refused on purpose"}})
    with pytest.raises(ResearchError) as e:
        providers.find("q", provider=f"fixture:{inj}")
    assert (e.value.kind, e.value.message) == ("blocked", "refused on purpose")


def test_fixture_list_form(write_json):
    path = write_json([{"url": "https://a.example", "title": "A"}])
    assert providers.find("q", provider=f"fixture:{path}")["results"][0]["rank"] == 1
