"""The built command, run as a subprocess, and flip driving it through its config."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flip_research import __version__

ROOT = Path(__file__).resolve().parent.parent
PAGE = ("<html><head><title>Tides</title></head><body><nav>Menu</nav><p>The harbour at Port "
        "Example floods twice a day, and the highest tide arrives in March.</p></body></html>")
FIXTURE = {
    "results": [
        {"url": "https://a.example/tides", "title": "Tides", "snippet": "harbour tides"},
        {"url": "https://a.example/tides/", "title": "Duplicate", "snippet": ""},
        {"url": "https://b.example/unreachable", "title": "Unreachable", "snippet": ""},
    ],
    "pages": {"https://a.example/tides": PAGE},
}


def command() -> list[str]:
    """The installed console script when present, else the module."""
    script = Path(sys.executable).parent / "flip-research"
    return [str(script)] if script.exists() else [sys.executable, "-m", "flip_research"]


def run(args, env_extra=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FLIP_RESEARCH", "BRAVE_", "TAVILY_", "ANTHROPIC_"))}
    env["PYTHONPATH"] = str(ROOT / "src")
    env.update(env_extra or {})
    return subprocess.run([*command(), *args], check=False, capture_output=True, text=True, env=env,
                          timeout=60)


@pytest.fixture
def fixture_env(write_json):
    return {"FLIP_RESEARCH_PROVIDER": f"fixture:{write_json(FIXTURE)}"}


def test_version():
    p = run(["--version"])
    assert p.returncode == 0 and __version__ in p.stdout


def test_find_prints_one_json_document(fixture_env):
    p = run(["find", "harbour tides", "--k", "5"], fixture_env)
    assert p.returncode == 0 and p.stderr == ""
    doc = json.loads(p.stdout)
    assert [r["rank"] for r in doc["results"]] == [1, 2]
    assert doc["query"] == "harbour tides" and doc["provider"] == "fixture"


def test_ask_extractive(fixture_env):
    p = run(["ask", "when does the harbour at Port Example flood"], fixture_env)
    assert p.returncode == 0, p.stderr
    doc = json.loads(p.stdout)
    assert doc["status"] == "answered" and doc["mode"] == "extractive"
    assert doc["citations"][0]["quote"].startswith("The harbour at Port Example floods")
    assert "Menu" not in p.stdout


def test_verify_without_model(fixture_env):
    p = run(["verify", "The harbour at Port Example floods twice a day"], fixture_env)
    doc = json.loads(p.stdout)
    assert doc["verdict"] == "uncertain" and "no model" in doc["reason"]


def test_error_is_one_json_line_on_stderr_and_nonzero(write_json):
    env = {"FLIP_RESEARCH_PROVIDER": "fixture:" + str(write_json(
        {"error": {"kind": "blocked", "message": "refused on purpose"}}))}
    p = run(["find", "q"], env)
    assert p.returncode == 1 and p.stdout == ""
    assert json.loads(p.stderr) == {"error": "refused on purpose", "error_kind": "blocked"}
    assert p.stderr.count("\n") == 1


def test_config_error_and_usage_error():
    p = run(["find", "q", "--provider", "brave"])
    assert p.returncode == 1 and json.loads(p.stderr)["error_kind"] == "config"
    p = run(["find"])
    assert p.returncode == 2 and json.loads(p.stderr)["error_kind"] == "usage"
    p = run(["find", "q", "--k", "0"])
    assert p.returncode == 1 and json.loads(p.stderr)["error_kind"] == "usage"


def test_half_configured_model_is_an_error_naming_the_variable(fixture_env):
    p = run(["ask", "anything"], {**fixture_env, "ANTHROPIC_API_KEY": "sk-test"})
    assert p.returncode == 1
    err = json.loads(p.stderr)
    assert err["error_kind"] == "config" and "FLIP_RESEARCH_MODEL" in err["error"]
    assert "sk-test" not in p.stderr


def test_doctor_reports_without_network():
    p = run(["doctor"], {"BRAVE_API_KEY": "k", "FLIP_RESEARCH_MODEL_URL": "http://localhost:1/v1",
                         "FLIP_RESEARCH_MODEL": "local-model"})
    doc = json.loads(p.stdout)
    assert doc["provider"]["selected"] == "brave"
    assert "BRAVE_API_KEY" in doc["provider"]["reason"]
    assert "BRAVE_API_KEY=k" not in p.stdout
    assert doc["model"]["mode"] == "model" and doc["model"]["model"] == "local-model"
    d2 = json.loads(run(["doctor"]).stdout)
    assert d2["provider"]["selected"] == "duckduckgo" and d2["model"]["mode"] == "extractive"


@pytest.mark.skipif(shutil.which("flip") is None, reason="flip is not installed")
def test_flip_accepts_the_output_and_records_leads(tmp_path, fixture_env):
    home, nb = tmp_path / "home", tmp_path / "nb"
    home.mkdir()
    cmd = " ".join(command())
    (home / "config.toml").write_text(
        f'[research]\nfind = "{cmd} find {{query}}"\nask = "{cmd} ask {{query}}"\n')
    env = {**os.environ, "FLIP_HOME": str(home), "PYTHONPATH": str(ROOT / "src"), **fixture_env}

    def flip(*args):
        return subprocess.run(["flip", *args], check=False, capture_output=True, text=True, env=env, cwd=nb,
                              timeout=120)

    new = subprocess.run(["flip", "new", "nb", "--kind", "ledger", "--dest", str(nb)],
                         check=False, capture_output=True, text=True, env=env, cwd=tmp_path)
    assert new.returncode == 0, new.stderr

    found = flip("find", "harbour tides", "--json")
    assert found.returncode == 0, found.stderr
    doc = json.loads(found.stdout)
    assert [c["url"] for c in doc["candidates"]] == [
        "https://a.example/tides", "https://b.example/unreachable"]
    assert doc["new_leads"] == 2
    leads = flip("lead", "list")
    assert "https://a.example/tides" in leads.stdout and "https://b.example/unreachable" in leads.stdout

    asked = flip("ask", "when does the harbour at Port Example flood", "--json")
    assert asked.returncode == 0, asked.stderr
    out = json.loads(asked.stdout)
    assert "floods twice a day" in out["answer"]
    assert [c["url"] for c in out["citations"]] == ["https://a.example/tides"]
    raw = json.loads((nb / out["raw"]).read_text())
    assert raw["status"] == "answered" and raw["citations"][0]["quote"]

    env["FLIP_RESEARCH_PROVIDER"] = "fixture:" + str(tmp_path / "missing.json")
    failed = flip("find", "harbour tides")
    assert failed.returncode != 0
