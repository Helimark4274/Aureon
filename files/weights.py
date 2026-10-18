"""Principles + adaptive weights via Hedge (multiplicative weights)."""
import math
from typing import Callable, Dict, Optional

from .types import Candidate

Principle = Callable[[Candidate, dict, dict], Optional[float]]   # -> [0,1], or None if uninformative here


def _tokens(s: str):
    return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if len(t) > 2}


def consensus(c: Candidate, ctx: dict, stats: dict) -> float:
    return stats["agreement"]


def grounding(c: Candidate, ctx: dict, stats: dict) -> float:
    """Share of answer tokens found in provided sources. None (excluded) when no sources are given."""
    sources = ctx.get("sources")
    if not sources:
        return None
    src = set().union(*[_tokens(s) for s in sources])
    toks = _tokens(c.text)
    return len(toks & src) / len(toks) if toks else 0.0


def brevity(c: Candidate, ctx: dict, stats: dict) -> float:
    """Penalise empty or runaway answers."""
    n = len(c.text.split())
    if n == 0:
        return 0.0
    return 1.0 if n <= 120 else max(0.0, 1 - (n - 120) / 400)


DEFAULT_PRINCIPLES: Dict[str, Principle] = {"consensus": consensus, "grounding": grounding, "brevity": brevity}


class AdaptiveWeights:
    """Multiplicative-weights update driven by *discrimination*, not by level.

    A naive Hedge loss |score - outcome| rewards any principle that always says "good" when the base
    accuracy is high (observed in testing: a constant principle took ~88% of the weight). So the update
    uses the running covariance between a principle's deviation from its own mean and the outcome's
    deviation from the base rate: principles that separate correct from wrong gain, constants stay flat.
    Principles that returned None for a decision are left out of combine() and update().
    """

    def __init__(self, names, eta: float = 2.0, floor: float = 0.05):
        self.eta, self.floor = eta, floor
        self.w = {n: 1 / len(names) for n in names}
        self._m = {n: 0.5 for n in names}   # running mean score per principle
        self._k = {n: 0 for n in names}
        self._ybar, self._ny = 0.5, 0

    def combine(self, scores: Dict[str, float]) -> float:
        z = sum(self.w[n] for n in scores)
        return sum(self.w[n] * s for n, s in scores.items()) / z

    def update(self, scores: Dict[str, float], correct: bool):
        y = 1.0 if correct else 0.0
        self._ny += 1
        self._ybar += (y - self._ybar) / self._ny
        mass = sum(self.w[n] for n in scores)
        for n, s in scores.items():
            self._k[n] += 1
            self._m[n] += (s - self._m[n]) / self._k[n]
            self.w[n] *= math.exp(self.eta * (s - self._m[n]) * (y - self._ybar))
        new_mass = sum(self.w[n] for n in scores)
        for n in scores:                      # keep the present principles' total mass unchanged
            self.w[n] *= mass / new_mass
        self.w = {n: max(v, self.floor) for n, v in self.w.items()}
        z = sum(self.w.values())
        self.w = {n: v / z for n, v in self.w.items()}
