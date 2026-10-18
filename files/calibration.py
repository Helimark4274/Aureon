"""Confidence calibration: Platt scaling + ECE/Brier. Pure Python."""
import math
from typing import Sequence

_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _sig(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def brier(conf: Sequence[float], correct: Sequence[int]) -> float:
    return sum((c - y) ** 2 for c, y in zip(conf, correct)) / max(len(conf), 1)


def reliability(conf: Sequence[float], correct: Sequence[int], bins: int = 10):
    """List of (lo, hi, n, mean_conf, accuracy)."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            out.append((lo, hi, 0, None, None))
            continue
        out.append((lo, hi, len(idx), sum(conf[i] for i in idx) / len(idx),
                    sum(correct[i] for i in idx) / len(idx)))
    return out


def ece(conf: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    n = len(conf)
    if n == 0:
        return 0.0
    return sum(nb / n * abs(mc - acc) for _, _, nb, mc, acc in reliability(conf, correct, bins) if nb)


def _loss(xs, ys, a, b, l2):
    tot = 0.0
    for x, y in zip(xs, ys):
        p = min(max(_sig(a * x + b), _EPS), 1 - _EPS)
        tot -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return tot + 0.5 * l2 * (a - 1.0) ** 2


class PlattCalibrator:
    """p = sigmoid(a * logit(conf) + b), fit by Newton's method with small L2 towards identity."""

    def __init__(self):
        self.a, self.b, self.fitted = 1.0, 0.0, False

    def fit(self, conf: Sequence[float], correct: Sequence[int], l2: float = 1e-3, iters: int = 50):
        if len(conf) < 30 or len(set(correct)) < 2:
            return self  # not enough signal: stay identity
        xs = [_logit(c) for c in conf]
        a, b = 1.0, 0.0
        for _ in range(iters):
            ga = gb = haa = hab = hbb = 0.0
            for x, y in zip(xs, correct):
                p = _sig(a * x + b)
                w = p * (1 - p)
                ga += (p - y) * x
                gb += (p - y)
                haa += w * x * x
                hab += w * x
                hbb += w
            ga += l2 * (a - 1.0)
            haa += l2
            hbb += 1e-9
            det = haa * hbb - hab * hab
            if abs(det) < 1e-12:
                break
            da = (ga * hbb - gb * hab) / det
            db = (haa * gb - hab * ga) / det
            # backtracking line search on regularised log-loss (plain Newton can diverge on saturated inputs)
            cur = _loss(xs, correct, a, b, l2)
            step = 1.0
            while step > 1e-6 and _loss(xs, correct, a - step * da, b - step * db, l2) > cur:
                step /= 2
            a, b = a - step * da, b - step * db
            if abs(step * da) + abs(step * db) < 1e-8:
                break
        self.a, self.b, self.fitted = a, b, True
        return self

    def __call__(self, conf: float) -> float:
        return _sig(self.a * _logit(conf) + self.b) if self.fitted else conf
