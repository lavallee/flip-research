# Changelog

## 0.1.0

- `find`, `ask`, `verify` and `doctor` commands, each printing one JSON document.
- Search providers: SearXNG, Brave, Tavily, and keyless best-effort DuckDuckGo HTML.
- Stdlib page fetcher with timeout, size cap, robots.txt, and a non-public-address guard.
- Optional model (OpenAI-compatible endpoint or Anthropic Messages API); every citation
  quote is verified verbatim against fetched text, and failures are dropped.
- Extractive mode when no model is configured.
