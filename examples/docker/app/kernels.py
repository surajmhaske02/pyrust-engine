"""Functions compiled ahead of time in the Docker build (see ../Dockerfile)."""
from pyrust_engine import optimize


@optimize
def word_counts(text: str) -> dict[str, int]:
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts


@optimize
def checksum(n: int) -> int:
    h = 2166136261
    for i in range(n):
        h = ((h ^ i) * 16777619) % 4294967296
    return h
