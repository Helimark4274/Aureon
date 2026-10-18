"""Aureon core: sample candidates -> score -> decide answer/abstain -> learn from outcomes."""
import re
from collections import Counter
from typing import Callable, Dict, List, Optional

from .calibration import PlattCalibrator
from .meta import MetaPatternDetector
from .store import Store
from .types import Candidate, Decision
from .weights import DEFAULT_PRINCIPLES, AdaptiveWeights

Generator = Callable[[str, dict, int], List[Candidate]]


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


class Aureon:
    def __init__(self, generator: Generator, n_candidates: int = 5, threshold: float = 0.7,
                 blind_spot_penalty: float = 0.1, store: Optional[Store] = None):
        self.generator = generator
        self.n = n_candidates
        self.threshold = threshold
        self.blind_spot_penalty = blind_spot_penalty
        self.principles = dict(DEFAULT_PRINCIPLES)
        self.weights = AdaptiveWeights(list(self.principles))
        self.calibrator = PlattCalibrator()
        self.detector = MetaPatternDetector()
        self.store = store or Store()
        self._pending: Dict[int, Decision] = {}
        self._next_id = 1
        self.history: List[tuple] = []   # (principle_scores, correct) for labelled decisions

    def answer(self, query: str, ctx: Optional[dict] = None) -> Decision:
        ctx = ctx or {}
        slice_key = ctx.get("slice", "default")
        cands = self.generator(query, ctx, self.n)
        did, self._next_id = self._next_id, self._next_id + 1
        thr = self.threshold + (self.blind_spot_penalty if self.detector.is_blind_spot(slice_key) else 0.0)

        if not cands:
            d = Decision(did, "abstain", None, 0.0, 0.0, 0.0, thr, slice_key, "no candidates generated",
                         reason_code="no_candidates", message="Modellen returnerede intet svar.")
        else:
            groups = Counter(normalize(c.text) for c in cands)
            best_score, best, best_scores, best_agree = -1.0, None, {}, 0.0
            for c in cands:
                agree = groups[normalize(c.text)] / len(cands)
                scores = {n: v for n, p in self.principles.items()
                          if (v := p(c, ctx, {"agreement": agree})) is not None}
                s = self.weights.combine(scores)
                if s > best_score:
                    best_score, best, best_scores, best_agree = s, c, scores, agree
            conf = self.calibrator(best_score)
            ok = conf >= thr
            d = Decision(did, "answer" if ok else "abstain", best.text if ok else None, best_score, conf,
                         best_agree, thr, slice_key,
                         "confidence above threshold" if ok else f"calibrated confidence {conf:.2f} < threshold {thr:.2f}",
                         best_scores, len(cands))
            d._shadow_text = best.text   # kept so feedback can judge the would-be answer
        self._explain(d)
        self._pending[did] = d
        self.store.log(d, query)
        if d.action == "abstain":
            self.store.log_gap(d, query)
        return d

    def _explain(self, d: Decision) -> None:
        """Turn silence into a structured 'not learned yet' / 'too uncertain' message."""
        d.labels_in_slice = self.detector.n.get(d.slice, 0)
        d.labels_needed = max(0, self.detector.min_n - d.labels_in_slice)
        if d.action == "answer":
            d.reason_code, d.message = "answered", ""
        elif d.reason_code == "no_candidates":
            pass
        elif d.labels_needed > 0:
            d.reason_code = "not_learned_yet"
            d.message = (f"Jeg kan ikke svare sikkert på dette endnu: jeg har kun {d.labels_in_slice} validerede "
                         f"eksempler inden for '{d.slice}' og har brug for mindst {self.detector.min_n}. "
                         f"Send flere rigtige/forkerte labels via /api/feedback.")
        else:
            d.reason_code = "low_confidence"
            d.message = (f"Jeg er for usikker på dette svar (kalibreret sikkerhed {d.confidence:.2f}, "
                         f"krav {d.threshold:.2f}), selvom jeg har {d.labels_in_slice} labels inden for '{d.slice}'.")

    def feedback(self, decision_id: int, correct: bool) -> None:
        """`correct` = was the best candidate correct (also for abstained decisions: shadow evaluation)."""
        d = self._pending.pop(decision_id, None)
        if d is None:
            return
        self.store.label(decision_id, correct)
        self.detector.record(d.slice, correct)
        if d.principle_scores:
            self.weights.update(d.principle_scores, correct)
            self.history.append((dict(d.principle_scores), int(correct)))

    def fit_calibration(self) -> None:
        """Refit on stored principle scores recombined with the CURRENT weights, so the calibrator sees the
        same raw-score distribution it will be applied to (weights drift during learning)."""
        if self.history:
            c = [self.weights.combine(sc) for sc, _ in self.history]
            y = [yy for _, yy in self.history]
            self.calibrator.fit(c, y)
