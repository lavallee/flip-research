import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests never see the developer's keys or provider choice."""
    for var in ("FLIP_RESEARCH_PROVIDER", "FLIP_RESEARCH_SEARXNG_URL", "BRAVE_API_KEY",
                "TAVILY_API_KEY", "FLIP_RESEARCH_MODEL_URL", "FLIP_RESEARCH_MODEL",
                "FLIP_RESEARCH_MODEL_KEY", "ANTHROPIC_API_KEY", "FLIP_RESEARCH_MODE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fixture_text():
    return lambda name: (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def write_json(tmp_path):
    def write(data, name="provider.json"):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return write
