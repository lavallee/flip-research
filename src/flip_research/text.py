"""A small stdlib HTML-to-text extractor: keep the paragraphs, drop the chrome."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

SKIP = {"script", "style", "nav", "header", "footer", "aside", "noscript", "form", "svg",
        "template", "iframe", "button", "select", "dialog"}
BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table",
         "section", "article", "main", "blockquote", "pre", "dd", "dt", "figcaption", "hr"}
VOID = {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "embed", "source",
        "track", "wbr"}

_WS = re.compile(r"\s+")


def norm_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass
class Extracted:
    title: str
    text: str  # paragraphs joined by a blank line


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._buf: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._title: list[str] = []

    def _flush(self) -> None:
        text = norm_ws("".join(self._buf))
        self._buf = []
        if text:
            self.paragraphs.append(text)

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in SKIP and tag not in VOID:
            self._skip_depth += 1
        if tag in BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in BLOCK:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        elif not self._skip_depth:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def extract(html: str) -> Extracted:
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    return Extracted(title=norm_ws("".join(parser._title)), text="\n\n".join(parser.paragraphs))
