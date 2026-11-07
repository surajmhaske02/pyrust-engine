"""Evaluate generated answers against references: exact match, token F1 and ROUGE-L.

Run:  python -m examples.eval_metrics

The metrics come from pyrust_engine.ai: compiled to Rust, with the same results as their
plain-Python source (shown in the timing comparison at the end).
"""

import random
import time

from pyrust_engine import ai

WORDS = 'the a model answer question paris france capital city river large small fast code python rust'.split()


def make_dataset(n, seed=0):
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        reference = ' '.join(rng.choice(WORDS) for _ in range(rng.randint(5, 30)))
        words = reference.split()
        prediction = ' '.join(w if rng.random() < 0.7 else rng.choice(WORDS) for w in words)
        data.append((prediction, reference))
    return data


def evaluate(pairs):
    n = len(pairs)
    return {
        'exact_match': sum(ai.exact_match(p, r) for p, r in pairs) / n,
        'token_f1': sum(ai.token_f1(p, r) for p, r in pairs) / n,
        'rouge_l': sum(ai.rouge_l(p, r) for p, r in pairs) / n,
    }


def main():
    pairs = make_dataset(5000)
    print(evaluate(pairs[:10]))  # first calls compile the metrics
    start = time.perf_counter()
    scores = evaluate(pairs)
    fast = time.perf_counter() - start
    start = time.perf_counter()
    plain = {
        'token_f1': sum(ai._token_f1(p, r) for p, r in pairs) / len(pairs),
        'rouge_l': sum(ai._rouge_l(p, r) for p, r in pairs) / len(pairs),
    }
    slow = time.perf_counter() - start
    assert plain['token_f1'] == scores['token_f1'] and plain['rouge_l'] == scores['rouge_l']
    print(scores)
    print(f'compiled metrics: {fast:.3f} s | plain Python (F1 + ROUGE-L only): {slow:.3f} s')


if __name__ == '__main__':
    main()
