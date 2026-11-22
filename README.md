# flip-research

`flip-research` is a small default for the `[research]` role of
[flip](https://github.com/lavallee/flip). flip lets you plug an external
command into `flip find` (candidate sources for a question) and `flip ask` (a
cited answer), but ships no default for either. This fills the role with public
pieces only: a web search, a plain page fetcher, and an optional language model
that you configure yourself.

It is a stub in the good sense: small, dependable, replaceable. Python 3.12+,
standard library only, no runtime dependencies.

## Install

```bash
uv tool install git+https://github.com/lavallee/flip-research
flip-research doctor
```

## Configure flip

Add two lines to `$FLIP_HOME/config.toml` (default `~/.flip/config.toml`):

```toml
[research]
find = "flip-research find {query}"
ask = "flip-research ask {query}"
```

Then `flip find "open knowledge format specification"` lists candidate sources
and records each as a lead; `flip ask "..."` saves a cited answer under
`sessions/raw/` as a grade-C lead. Neither captures a source: do that yourself
with `flip add-source` on the URLs worth keeping.

## Commands

Each prints one JSON document on stdout and exits 0. On failure it exits
non-zero with one JSON line on stderr: `{"error": "...", "error_kind": "..."}`.

| Command | Output |
|---|---|
| `find QUERY [--k N] [--provider NAME]` | `{"results":[{"url","title","snippet","provider","rank"}], "query", "provider", "did":[...]}` |
| `ask QUESTION [--k N]` | `{"answer", "citations":[{"url","title","quote"}], "status":"answered\|not_found\|unavailable", "mode", "did":[...]}` |
| `verify CLAIM [--k N]` | `{"verdict":"supported\|refuted\|uncertain", "citations":[...], "reason", "did":[...]}` |
| `doctor` | which provider and model would be used, and why. No network call, no paid call. |

`did` is a plain-language log of what happened: what was searched, which pages
were fetched or skipped (and why), which citations were dropped.

## Search providers

Chosen by `--provider` or `FLIP_RESEARCH_PROVIDER`; otherwise the first one
configured, in this order. There is no silent fallback: if the chosen provider
fails, the command fails and says so.

| Provider | Enabled by | Trade-offs |
|---|---|---|
| `searxng` | `FLIP_RESEARCH_SEARXNG_URL` (an instance with the JSON format enabled) | Self-hosted, no key, you control it. Many public instances disable the JSON API and answer 403. |
| `brave` | `BRAVE_API_KEY` | Independent index, stable API, free tier is small. |
| `tavily` | `TAVILY_API_KEY` | Built for agents, returns content snippets; a hosted service with its own terms. |
| `duckduckgo` | nothing (the default when nothing else is set) | **Keyless and best effort.** It parses DuckDuckGo's HTML endpoint, which is not an API: it may be refused, rate limited or change shape without notice. When it refuses, `find` exits non-zero with `error_kind: "blocked"` and never returns invented results. For regular use, set one of the providers above. |

Results are de-duplicated by normalised URL (host case, fragments, default
ports, trailing slashes and tracking parameters ignored).

## Asking and verifying

`ask` runs `find`, fetches the top results (standard-library `urllib`; 15 second
timeout; 1 MB cap; `User-Agent: flip-research/<version>` with no contact
details; `robots.txt` obeyed; non-public addresses refused), and extracts the
readable text: scripts, styles and navigation dropped, paragraphs kept. HTML and
plain text only; PDFs are skipped.

- **With a model**, it is asked to answer only from the fetched passages, with a
  citation (an exact quotation) for every sentence, and to say so when the
  passages do not answer. Afterwards every quote is checked against the fetched
  text of the page it cites. A citation that does not appear word for word is
  dropped, and a sentence left with no verified citation is dropped with it.
- **Without a model**, it returns the best-matching passages verbatim, with their
  URLs, and `"mode": "extractive"`. No synthesis happens.

`verify` retrieves the same way. With a model it returns `supported`,
`refuted` or `uncertain` plus verified quotes; a `supported` or `refuted`
verdict without a surviving quote is downgraded to `uncertain`. Without a model
the verdict is always `uncertain`, with the passages and the reason.

### Configuring a model

No default model name is built in; you must set `FLIP_RESEARCH_MODEL`.

```bash
# Any OpenAI-compatible chat-completions endpoint, including local servers
export FLIP_RESEARCH_MODEL_URL=http://localhost:11434/v1   # base URL or full .../chat/completions
export FLIP_RESEARCH_MODEL=<model name your server knows>
export FLIP_RESEARCH_MODEL_KEY=<key, if the endpoint needs one>

# Or Anthropic's Messages API over plain HTTPS
export ANTHROPIC_API_KEY=...
export FLIP_RESEARCH_MODEL=<a Claude model name>
```

If `ANTHROPIC_API_KEY` is present for other reasons and you want no model here,
set `FLIP_RESEARCH_MODE=extractive`. A half-configured model (a URL or key with
no model name) is an error, not a silent fallback. Passages sent to a model leave
your machine; the paid call happens only in `ask` and `verify`, never in `doctor`.

## Honesty rules

1. **Nothing is invented.** Every result comes from a provider's response; if
   there are none, there are none.
2. **Quotes are verified.** A citation is kept only if its quote occurs verbatim
   (whitespace aside, and at least 12 characters) in text this run fetched.
3. **A synthesis is a lead, not a source.** A model can still misread a real
   quote. Treat `ask` output as a pointer to pages to read, capture and grade in
   flip, not as evidence.
4. **Absence is reported as absence.** `not_found` and `unavailable` are
   distinct from an answer, and `did` says why.

Fetched pages are untrusted input. The model is told to ignore instructions
inside them, but a quote check cannot detect a page that lies; it only proves
the page said it.

## Development

```bash
uv sync
uv run ruff check
uv run pytest -q
```

Tests are offline. Providers and the model are plain functions that tests
replace. `FLIP_RESEARCH_PROVIDER=fixture:<path.json>` is a test hook that serves
results (and pages for `ask`) from a file; see `tests/fixtures/`.

## Licence

MIT.
