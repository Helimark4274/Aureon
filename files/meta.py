"""Blind-spot detection: which slices (topic/context) have a significantly higher error rate?"""
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


class MetaPatternDetector:
    def __init__(self, min_n: int = 20):
        self.min_n = min_n
        self.err: Dict[str, int] = defaultdict(int)
        self.n: Dict[str, int] = defaultdict(int)

    def record(self, slice_key: str, correct: bool):
        self.n[slice_key] += 1
        self.err[slice_key] += 0 if correct else 1

    @property
    def overall_error(self) -> float:
        tot = sum(self.n.values())
        return sum(self.err.values()) / tot if tot else 0.0

    def report(self) -> List[dict]:
        base = self.overall_error
        rows = []
        for s, n in self.n.items():
            k = self.err[s]
            lo, hi = wilson_interval(k, n)
            rows.append(dict(slice=s, n=n, error_rate=k / n, ci_low=lo, ci_high=hi,
                             flagged=(n >= self.min_n and lo > base)))
        return sorted(rows, key=lambda r: -r["error_rate"])

    def is_blind_spot(self, slice_key: str) -> bool:
        n = self.n.get(slice_key, 0)
        if n < self.min_n:
            return False
        lo, _ = wilson_interval(self.err[slice_key], n)
        return lo > self.overall_error
