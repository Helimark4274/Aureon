"""Run a labelled CSV (question,truth,topic) through Aureon and report honest numbers."""
import csv
import random
from collections import defaultdict
from typing import Dict, List

from .calibration import brier, ece
from .engine import Aureon, normalize
from .meta import wilson_interval


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("question") or "").strip() and (r.get("truth") or "").strip()]
    for r in rows:
        r["topic"] = (r.get("topic") or "default").strip() or "default"
    return rows


def is_correct(answer: str, truth: str, exact: bool = False) -> bool:
    a, t = normalize(answer), normalize(truth)
    return a == t if exact else (t in a)


def run(rows, generator, train_frac=0.5, seed=0, threshold=0.7, n_candidates=5, exact=False, pass_truth=False):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    k = int(len(rows) * train_frac)
    train, test = rows[:k], rows[k:]
    a = Aureon(generator, n_candidates=n_candidates, threshold=threshold)

    def ctx_of(r):
        c = {"slice": r["topic"]}
        if pass_truth:
            c["truth"] = r["truth"]
        return c

    for r in train:                                        # phase 1: learn from labels
        d = a.answer(r["question"], ctx_of(r))
        a.feedback(d.id, is_correct(d._shadow_text, r["truth"], exact) if hasattr(d, "_shadow_text") else False)
    a.fit_calibration()

    res = []                                               # phase 2: frozen test
    for r in test:
        d = a.answer(r["question"], ctx_of(r))
        ok = is_correct(d._shadow_text, r["truth"], exact) if hasattr(d, "_shadow_text") else False
        res.append((d, ok, r["topic"]))
    return a, train, test, res


def report(a, train, test, res, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)) -> str:
    y = [int(ok) for _, ok, _ in res]
    cal = [d.confidence for d, _, _ in res]
    raw = [d.raw_confidence for d, _, _ in res]
    n = len(y)
    out = ["# Aureon – evaluering på dine data\n",
           f"Træning (med labels): {len(train)} · Test (frosset): {n}\n"]
    if n < 100:
        out.append(f"> ADVARSEL: kun {n} testspørgsmål. Konfidensintervallerne nedenfor er brede; tag ikke konklusioner for givet.\n")
    lo, hi = wilson_interval(sum(y), n)
    out.append(f"Baseline (svar altid): accuracy **{sum(y)/max(n,1):.3f}** (95% CI [{lo:.2f}, {hi:.2f}])\n")
    out.append("## Kalibrering\n\n| Mål | Rå | Kalibreret |\n|---|---|---|")
    out.append(f"| ECE | {ece(raw, y):.3f} | {ece(cal, y):.3f} |")
    out.append(f"| Brier | {brier(raw, y):.3f} | {brier(cal, y):.3f} |\n")
    out.append("## Afstå-kurve (test)\n\n| Tærskel | Coverage | Besvarede | Accuracy | 95% CI | Fejl undgået |\n|---|---|---|---|---|---|")
    base_err = n - sum(y)
    for t in thresholds:
        idx = [i for i, c in enumerate(cal) if c >= t]
        if not idx:
            out.append(f"| {t} | 0.000 | 0 | – | – | – |"); continue
        k = sum(y[i] for i in idx)
        lo, hi = wilson_interval(k, len(idx))
        out.append(f"| {t} | {len(idx)/n:.3f} | {len(idx)} | {k/len(idx):.3f} | [{lo:.2f}, {hi:.2f}] | {base_err-(len(idx)-k)} af {base_err} |")
    out.append("\n## Emner (træningslabels)\n\n| Emne | n | Fejlrate | 95% CI | Flagget |\n|---|---|---|---|---|")
    for r in a.detector.report():
        out.append(f"| {r['slice']} | {r['n']} | {r['error_rate']:.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {'ja' if r['flagged'] else 'nej'} |")
    out.append("\n## Afståelser fordelt på årsag og emne (hele kørslen)\n\n| Emne | Årsag | Antal |\n|---|---|---|")
    for s, rc, c in a.store.gaps():
        out.append(f"| {s} | {rc} | {c} |")
    out.append("\n**Sådan læser du det:** Hjælper afståelse, stiger *Accuracy* ved højere tærskel, mens *Coverage* falder. "
               "Er det ikke tilfældet (eller overlapper CI'erne med baseline), gør Aureon ingen forskel på dine data.")
    return "\n".join(out)
