"""A FastAPI service with CPU-heavy endpoints compiled by PyRust.

Run:  pip install fastapi uvicorn "pyrust-engine[arrays]"
      uvicorn examples.fastapi_app:app

Compiled calls don't hold Python's GIL, so running them with asyncio.to_thread keeps
the server answering other requests while one computes.
"""

import asyncio

from fastapi import FastAPI

from pyrust_engine import optimize
from pyrust_engine.ai import BM25, levenshtein, token_f1

app = FastAPI(title='PyRust demo')

CATALOG = [
    'wireless noise cancelling headphones',
    'mechanical keyboard with brown switches',
    'usb-c charging cable two metres',
    'ergonomic office chair with lumbar support',
    'noise cancelling earbuds with charging case',
]
INDEX = BM25(CATALOG)


@optimize
def spam_score(text: str) -> float:
    """A hand-written scoring rule: compiled because it loops over characters and words."""
    upper = 0
    letters = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if ch.isupper():
                upper += 1
    words = text.split()
    repeats = len(words) - len(set(words))
    shouting = upper / letters if letters else 0.0
    return min(1.0, shouting * 0.7 + repeats * 0.05 + text.count('!') * 0.1)


@app.get('/search')
async def search(q: str, k: int = 3):
    ids, scores = await asyncio.to_thread(INDEX.search, q, k)
    return [{'item': CATALOG[i], 'score': float(s)} for i, s in zip(ids.tolist(), scores.tolist()) if s > 0]


@app.get('/similarity')
async def similarity(a: str, b: str):
    distance = await asyncio.to_thread(levenshtein, a, b)
    return {'edit_distance': distance, 'token_f1': token_f1(a, b)}


@app.get('/spam')
async def spam(text: str):
    return {'score': await asyncio.to_thread(spam_score, text)}
