from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def sentence_count(text: str) -> int:
    text = (text or "").strip()
    return 0 if not text else len([part for part in SENTENCE_RE.split(text) if part.strip()])


def longest_character_run(text: str) -> int:
    if not text:
        return 0
    longest = current = 1
    for previous, character in zip(text, text[1:]):
        current = current + 1 if character == previous else 1
        longest = max(longest, current)
    return longest


def bigram_repetition(text: str) -> float:
    words = [word.lower() for word in WORD_RE.findall(text or "")]
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def content_unstable(text: str) -> bool:
    return (
        word_count(text) == 0
        or longest_character_run(text) >= 20
        or bigram_repetition(text) >= 0.35
    )

