"""README fetching and markdown cleaning for Osiris ingestion payloads."""

from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass
from typing import Any


_BADGE_RE = re.compile(r"!?\[[^\]]*\]\([^)]*(?:badge|shields\.io|badge\.svg)[^)]*\)", re.I)
_IMAGE_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\.(?:png|jpg|jpeg|gif|svg|webp)(?:\?[^)]*)?\)", re.I)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(table of contents|toc)\s*$", re.I),
    re.compile(r"^\s*(license|copyright)\s*$", re.I),
    re.compile(r"^\s*(built with|powered by)\s*$", re.I),
    re.compile(r"^\s*(npm install|pip install|cargo install|go get)\b", re.I),
]


@dataclass(slots=True)
class ReadmeDocument:
    raw_markdown: str
    clean_text: str
    extracted_paragraphs: list[str]
    readme_length: int
    readme_markdown: str = ""


def decode_readme_payload(payload: dict[str, Any] | None) -> str:
    """Decode a GitHub README API payload into markdown text."""
    if not payload:
        return ""
    content = payload.get("content") or ""
    encoding = (payload.get("encoding") or "").lower()
    if encoding == "base64":
        try:
            return base64.b64decode(content, validate=False).decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(content, str):
        return content
    return ""


def process_readme_payload(payload: dict[str, Any] | None) -> ReadmeDocument:
    return process_markdown(decode_readme_payload(payload))


def process_markdown(markdown: str) -> ReadmeDocument:
    """Clean markdown and extract paragraphs useful to Osiris semantic stages."""
    raw = markdown or ""
    text = html.unescape(raw)
    text = _BADGE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _FENCE_RE.sub(" ", text)
    text = _MARKDOWN_TABLE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n+", text):
        block = re.sub(r"\s+", " ", block).strip(" -*_\t\n")
        if not _is_meaningful_paragraph(block):
            continue
        paragraphs.append(block)

    clean_text = "\n\n".join(paragraphs)
    return ReadmeDocument(
        raw_markdown=raw,
        clean_text=clean_text,
        extracted_paragraphs=paragraphs[:80],
        readme_length=len(raw),
        readme_markdown="",
    )


def _is_meaningful_paragraph(text: str) -> bool:
    if len(text) < 40:
        return False
    if len(text.split()) < 6:
        return False
    if any(pattern.search(text) for pattern in _BOILERPLATE_PATTERNS):
        return False
    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars / max(len(text), 1) < 0.45:
        return False
    return True
