"""Tiny optional helpers for reading the task knowledge base."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"


def list_knowledge() -> list[str]:
    return sorted(path.name for path in KNOWLEDGE_DIR.glob("*.md"))


def read_knowledge(name: str) -> str:
    path = (KNOWLEDGE_DIR / name).resolve()
    if KNOWLEDGE_DIR.resolve() not in path.parents:
        raise ValueError("knowledge path must stay inside knowledge/")
    return path.read_text()


def search_knowledge(query: str) -> list[tuple[str, str]]:
    terms = [term.lower() for term in query.split() if term.strip()]
    matches: list[tuple[str, str]] = []
    for path in KNOWLEDGE_DIR.glob("*.md"):
        for line in path.read_text().splitlines():
            haystack = line.lower()
            if terms and all(term in haystack for term in terms):
                matches.append((path.name, line))
    return matches
