"""Optional language model, configured only by environment and called over stdlib HTTPS.

No default model name is built in: the user names the model, or there is no model and
`ask` falls back to quoting passages. Tests replace `call_model`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import net
from .errors import ResearchError

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 1500
MODEL_TIMEOUT = 120.0


@dataclass(frozen=True)
class ModelConfig:
    kind: str  # "openai" (any OpenAI-compatible chat-completions endpoint) or "anthropic"
    model: str
    url: str
    key: str = ""

    def label(self) -> str:
        return f"{self.model} via {'Anthropic Messages API' if self.kind == 'anthropic' else self.url}"


def configure(env=None) -> ModelConfig | None:
    """The configured model, or None for extractive mode. Raises a `config` error when
    the environment half-configures a model (for instance, no model name)."""
    env = os.environ if env is None else env
    if env.get("FLIP_RESEARCH_MODE") == "extractive":
        return None
    name = (env.get("FLIP_RESEARCH_MODEL") or "").strip()
    url = (env.get("FLIP_RESEARCH_MODEL_URL") or "").strip()
    if url:
        if not name:
            raise ResearchError(
                "config", "FLIP_RESEARCH_MODEL_URL is set but FLIP_RESEARCH_MODEL is not; "
                "name the model to use (no default is built in)")
        url = url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        return ModelConfig("openai", name, url, env.get("FLIP_RESEARCH_MODEL_KEY", ""))
    if env.get("ANTHROPIC_API_KEY"):
        if not name:
            raise ResearchError(
                "config", "ANTHROPIC_API_KEY is set but FLIP_RESEARCH_MODEL is not; name a "
                "Claude model (no default is built in), or set FLIP_RESEARCH_MODE=extractive "
                "to run without a model")
        return ModelConfig("anthropic", name, ANTHROPIC_URL, env["ANTHROPIC_API_KEY"])
    if name:
        raise ResearchError(
            "config", "FLIP_RESEARCH_MODEL is set but there is nothing to call: set "
            "FLIP_RESEARCH_MODEL_URL (OpenAI-compatible) or ANTHROPIC_API_KEY")
    return None


def describe(env=None) -> dict:
    """For `doctor`: what would be used and why. Makes no network call."""
    env = os.environ if env is None else env
    try:
        cfg = configure(env)
    except ResearchError as exc:
        return {"mode": "error", "error_kind": exc.kind, "reason": exc.message}
    if cfg is None:
        if env.get("FLIP_RESEARCH_MODE") == "extractive":
            why = "FLIP_RESEARCH_MODE=extractive"
        else:
            why = "no model configured"
        return {"mode": "extractive", "reason": f"{why}; ask quotes passages verbatim and "
                "verify always answers uncertain"}
    return {"mode": "model", "model": cfg.model, "api": cfg.kind, "reason": cfg.label()}


def call_model(cfg: ModelConfig, system: str, user: str) -> str:
    """One chat turn; returns the reply text. Raises ResearchError(kind="model")."""
    if cfg.kind == "anthropic":
        body = {"model": cfg.model, "max_tokens": MAX_TOKENS, "system": system,
                "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": cfg.key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
    else:
        body = {"model": cfg.model, "temperature": 0, "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        headers = {"content-type": "application/json"}
        if cfg.key:
            headers["authorization"] = f"Bearer {cfg.key}"
    resp = net.http_get(cfg.url, data=json.dumps(body).encode(), headers=headers,
                        timeout=MODEL_TIMEOUT, max_bytes=4_000_000)
    if resp.status != 200:
        raise ResearchError("model", f"the model endpoint answered HTTP {resp.status}")
    try:
        data = json.loads(net.decode(resp))
        if cfg.kind == "anthropic":
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        return data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ResearchError("model", "the model endpoint returned an unexpected response") from exc
