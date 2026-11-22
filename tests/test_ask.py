import json

import pytest

from flip_research import ask, model, net
from flip_research.ask import Doc
from flip_research.errors import ResearchError

QUESTION = "when does the harbour at Port Example flood"
FLOOD = "The harbour at Port Example floods twice a day, and the highest tide arrives in March."
OTHER = "Fishing boats leave before dawn and most return by noon with cod and haddock."
URL1, URL2, URL3 = "https://a.example/tides", "https://b.example/boats", "https://c.example/x"


@pytest.fixture
def provider(write_json):
    path = write_json({"results": [
        {"url": URL1, "title": "Tides", "snippet": ""},
        {"url": URL2, "title": "Boats", "snippet": ""},
        {"url": URL3, "title": "Unreachable", "snippet": ""},
    ]})
    return f"fixture:{path}"


def fetcher(pages):
    def fetch(url):
        if url not in pages:
            raise ResearchError("http", "HTTP 404")
        return Doc(url, "", pages[url])

    return fetch


PAGES = {URL1: f"Intro paragraph that is unrelated to the matter at hand.\n\n{FLOOD}",
         URL2: OTHER}


def set_model(monkeypatch, reply):
    monkeypatch.setenv("FLIP_RESEARCH_MODEL_URL", "http://localhost:1/v1")
    monkeypatch.setenv("FLIP_RESEARCH_MODEL", "test-model")
    calls = []

    def fake(cfg, system, user):
        calls.append((cfg, system, user))
        return reply if isinstance(reply, str) else json.dumps(reply)

    monkeypatch.setattr(model, "call_model", fake)
    return calls


# --- extractive -----------------------------------------------------------------------

def test_extractive_quotes_best_passages_verbatim(provider):
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "answered" and out["mode"] == "extractive"
    assert out["citations"][0] == {"url": URL1, "title": "Tides", "quote": FLOOD}
    assert URL1 in out["answer"] and FLOOD in out["answer"]
    for c in out["citations"]:
        assert ask.quote_is_verbatim(c["quote"], PAGES[c["url"]])
    assert any("skipped " + URL3 in d for d in out["did"])
    assert any("no model configured" in d for d in out["did"])


def test_extractive_not_found_when_nothing_matches(provider):
    out = ask.ask("quantum chromodynamics lattice", provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "not_found" and out["citations"] == [] and out["answer"] == ""


def test_unavailable_when_no_page_can_be_read(provider):
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher({}))
    assert out["status"] == "unavailable"
    assert sum("skipped" in d for d in out["did"]) == 3


def test_not_found_when_search_is_empty(write_json):
    path = write_json({"results": []})
    out = ask.ask(QUESTION, provider=f"fixture:{path}", fetcher=fetcher(PAGES))
    assert out["status"] == "not_found"


def test_empty_question():
    with pytest.raises(ResearchError) as e:
        ask.ask("  ", provider="duckduckgo")
    assert e.value.kind == "usage"


# --- model synthesis and quote verification -------------------------------------------

GOOD = {"answerable": True, "sentences": [
    {"text": "The harbour floods twice a day.",
     "citations": [{"source": "S1", "quote": "floods twice a day"}]},
    {"text": "The biggest tide comes in March.",
     "citations": [{"source": "S1", "quote": "the highest tide arrives in March"}]},
]}


def test_model_answer_keeps_verified_citations(monkeypatch, provider):
    calls = set_model(monkeypatch, GOOD)
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "answered" and out["mode"] == "synthesis"
    assert out["model"] == "test-model"
    assert out["answer"] == ("The harbour floods twice a day. [1] "
                             "The biggest tide comes in March. [2]")
    assert [c["quote"] for c in out["citations"]] == [
        "floods twice a day", "the highest tide arrives in March"]
    assert calls[0][0].url == "http://localhost:1/v1/chat/completions"
    assert "ONLY" in calls[0][1] and "untrusted" in calls[0][1]
    assert FLOOD in calls[0][2]


def test_fabricated_quote_loses_its_citation_and_sentence(monkeypatch, provider):
    reply = {"answerable": True, "sentences": [
        GOOD["sentences"][0],
        {"text": "Harbour fees double in summer.",
         "citations": [{"source": "S1", "quote": "fees double during the summer season"}]},
        {"text": "Mixed support.",
         "citations": [{"source": "S2", "quote": "a quote nobody ever wrote"},
                       {"source": "S2", "quote": "most return by noon"}]},
    ]}
    set_model(monkeypatch, reply)
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    quotes = [c["quote"] for c in out["citations"]]
    assert quotes == ["floods twice a day", "most return by noon"]
    assert "fees double" not in out["answer"]
    assert "Mixed support. [2]" in out["answer"]
    assert any("not found verbatim" in d and "fees double" in d for d in out["did"])
    assert any("dropped 1 sentence" in d for d in out["did"])


def test_all_fabricated_means_not_found(monkeypatch, provider):
    set_model(monkeypatch, {"answerable": True, "sentences": [
        {"text": "Invented.", "citations": [{"source": "S1", "quote": "entirely made up text"}]}]})
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "not_found" and out["citations"] == [] and out["answer"] == ""


def test_quote_must_come_from_the_cited_source(monkeypatch, provider):
    set_model(monkeypatch, {"answerable": True, "sentences": [
        {"text": "Wrong page.", "citations": [{"source": "S2", "quote": "floods twice a day"}]}]})
    assert ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))["status"] == "not_found"


def test_unknown_source_and_short_quote_dropped(monkeypatch, provider):
    set_model(monkeypatch, {"answerable": True, "sentences": [
        {"text": "A.", "citations": [{"source": "S9", "quote": "floods twice a day"}]},
        {"text": "B.", "citations": [{"source": "S1", "quote": "the"}]}]})
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "not_found"
    assert any("unknown source" in d for d in out["did"])


def test_model_says_passages_do_not_answer(monkeypatch, provider):
    set_model(monkeypatch, {"answerable": False, "sentences": []})
    out = ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "not_found"
    assert any("not to answer" in d for d in out["did"])


def test_fenced_json_reply_is_accepted(monkeypatch, provider):
    set_model(monkeypatch, "```json\n" + json.dumps(GOOD) + "\n```")
    assert ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))["status"] == "answered"


def test_non_json_reply_is_an_error(monkeypatch, provider):
    set_model(monkeypatch, "The harbour floods twice a day.")
    with pytest.raises(ResearchError) as e:
        ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert e.value.kind == "model_output"


def test_model_not_called_when_nothing_matches(monkeypatch, provider):
    calls = set_model(monkeypatch, GOOD)
    out = ask.ask("quantum chromodynamics lattice", provider=provider, fetcher=fetcher(PAGES))
    assert out["status"] == "not_found" and calls == []


def test_model_failure_surfaces(monkeypatch, provider):
    set_model(monkeypatch, "x")

    def boom(*a):
        raise ResearchError("model", "the model endpoint answered HTTP 500")

    monkeypatch.setattr(model, "call_model", boom)
    with pytest.raises(ResearchError) as e:
        ask.ask(QUESTION, provider=provider, fetcher=fetcher(PAGES))
    assert e.value.kind == "model"


def test_quote_is_verbatim_ignores_only_whitespace():
    assert ask.quote_is_verbatim("floods   twice\na day", "it floods twice a day.")
    assert not ask.quote_is_verbatim("Floods twice a day", "it floods twice a day.")
    assert not ask.quote_is_verbatim("a day", "it floods twice a day.")  # too short


# --- verify ---------------------------------------------------------------------------

def test_verify_without_model_is_always_uncertain(provider):
    out = ask.verify("The harbour at Port Example floods twice a day", provider=provider,
                     fetcher=fetcher(PAGES))
    assert out["verdict"] == "uncertain" and out["mode"] == "extractive"
    assert "no model is configured" in out["reason"]
    assert out["citations"][0]["quote"] == FLOOD


def test_verify_supported_with_verified_quote(monkeypatch, provider):
    set_model(monkeypatch, {"verdict": "supported", "reason": "Stated directly.",
                            "citations": [{"source": "S1", "quote": "floods twice a day"}]})
    out = ask.verify("The harbour floods twice a day", provider=provider, fetcher=fetcher(PAGES))
    assert out["verdict"] == "supported" and out["reason"] == "Stated directly."
    assert out["citations"] == [{"url": URL1, "title": "Tides", "quote": "floods twice a day"}]


def test_verify_downgrades_when_quote_fabricated(monkeypatch, provider):
    set_model(monkeypatch, {"verdict": "refuted", "reason": "Says otherwise.",
                            "citations": [{"source": "S1", "quote": "never floods at all"}]})
    out = ask.verify("The harbour floods twice a day", provider=provider, fetcher=fetcher(PAGES))
    assert out["verdict"] == "uncertain" and out["citations"] == []
    assert any("downgraded" in d for d in out["did"])


def test_verify_bad_verdict_word(monkeypatch, provider):
    set_model(monkeypatch, {"verdict": "probably", "reason": "x", "citations": []})
    out = ask.verify("The harbour floods", provider=provider, fetcher=fetcher(PAGES))
    assert out["verdict"] == "uncertain"


def test_verify_no_pages_and_empty_claim(provider):
    out = ask.verify("The harbour floods", provider=provider, fetcher=fetcher({}))
    assert out["verdict"] == "uncertain" and "could be fetched" in out["reason"]
    with pytest.raises(ResearchError):
        ask.verify(" ", provider=provider)


# --- model configuration --------------------------------------------------------------

def test_model_config_errors_and_modes():
    with pytest.raises(ResearchError) as e:
        model.configure({"FLIP_RESEARCH_MODEL_URL": "http://x/v1"})
    assert e.value.kind == "config" and "FLIP_RESEARCH_MODEL" in e.value.message
    with pytest.raises(ResearchError) as e:
        model.configure({"ANTHROPIC_API_KEY": "k"})
    assert "name a Claude model" in e.value.message
    with pytest.raises(ResearchError):
        model.configure({"FLIP_RESEARCH_MODEL": "m"})
    assert model.configure({}) is None
    assert model.configure({"ANTHROPIC_API_KEY": "k", "FLIP_RESEARCH_MODE": "extractive"}) is None
    cfg = model.configure({"FLIP_RESEARCH_MODEL_URL": "http://x/v1/chat/completions/",
                           "FLIP_RESEARCH_MODEL": "m", "FLIP_RESEARCH_MODEL_KEY": "k"})
    assert cfg.url == "http://x/v1/chat/completions" and cfg.key == "k"
    assert model.configure({"ANTHROPIC_API_KEY": "k", "FLIP_RESEARCH_MODEL": "m"}).kind == \
        "anthropic"
    assert model.describe({"ANTHROPIC_API_KEY": "k"})["mode"] == "error"


def test_ask_fails_before_fetching_when_model_half_configured(monkeypatch, provider):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def no_fetch(url):
        raise AssertionError("fetched")

    with pytest.raises(ResearchError) as e:
        ask.ask(QUESTION, provider=provider, fetcher=no_fetch)
    assert e.value.kind == "config"


def _capture(monkeypatch, body):
    seen = {}

    def fake(url, **kw):
        seen.update(url=url, **kw)
        return net.Response(status=200, url=url, body=json.dumps(body).encode())

    monkeypatch.setattr(net, "http_get", fake)
    return seen


def test_anthropic_request_shape(monkeypatch):
    seen = _capture(monkeypatch, {"content": [{"type": "text", "text": "hello"}]})
    cfg = model.ModelConfig("anthropic", "claude-x", model.ANTHROPIC_URL, "sk-test")
    assert model.call_model(cfg, "SYS", "USER") == "hello"
    body = json.loads(seen["data"])
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"] == {"x-api-key": "sk-test", "anthropic-version": "2023-06-01",
                               "content-type": "application/json"}
    assert body["model"] == "claude-x" and body["system"] == "SYS" and body["max_tokens"] > 0
    assert body["messages"] == [{"role": "user", "content": "USER"}]


def test_openai_request_shape(monkeypatch):
    seen = _capture(monkeypatch, {"choices": [{"message": {"content": "hi"}}]})
    cfg = model.ModelConfig("openai", "m", "http://localhost:1/v1/chat/completions", "k")
    assert model.call_model(cfg, "SYS", "USER") == "hi"
    assert seen["headers"]["authorization"] == "Bearer k"
    assert json.loads(seen["data"])["messages"][0] == {"role": "system", "content": "SYS"}
    cfg = model.ModelConfig("openai", "m", "http://localhost:1/v1/chat/completions")
    model.call_model(cfg, "S", "U")
    assert "authorization" not in seen["headers"]


def test_model_http_and_shape_errors(monkeypatch):
    cfg = model.ModelConfig("openai", "m", "http://x/chat/completions")
    monkeypatch.setattr(net, "http_get", lambda *a, **k: net.Response(500, "u"))
    with pytest.raises(ResearchError) as e:
        model.call_model(cfg, "s", "u")
    assert e.value.kind == "model" and "500" in e.value.message
    monkeypatch.setattr(net, "http_get",
                        lambda *a, **k: net.Response(200, "u", body=b'{"oops": 1}'))
    with pytest.raises(ResearchError) as e:
        model.call_model(cfg, "s", "u")
    assert e.value.kind == "model"


# --- passage helpers ------------------------------------------------------------------

def test_split_passages_breaks_long_paragraphs_on_sentences():
    long = " ".join(f"Sentence number {i} says something useful." for i in range(40))
    parts = ask.split_passages(long, limit=200)
    assert all(len(p) <= 200 for p in parts) and len(parts) > 3
    assert " ".join(parts) == long


def test_rank_prefers_rarer_terms_and_caps_per_doc():
    docs = [Doc("u1", "", "tides tides tides are regular and predictable today.\n\n"
                          "the harbour floods twice daily at spring tide time."),
            Doc("u2", "", "harbour fees are listed in the annual harbour notice here.")]
    top = ask.rank("harbour floods", docs, 5)
    assert top[0].doc == 0 and "floods" in top[0].text
    assert top[0].coverage == 1.0
