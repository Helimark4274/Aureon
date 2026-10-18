"""End-to-end evaluation on a SYNTHETIC model with known per-topic accuracy.
Proves the machinery (calibration, abstention, blind-spot detection, weight adaptation) works;
says nothing about any real LLM until you plug one in (see aureon/llm_anthropic.py)."""
from collections import defaultdict
from aureon import Aureon, ece, brier
from aureon.sim import make_generator, question_stream, TOPIC_ACC

N_TRAIN, N_TEST = 1500, 3000
a = Aureon(make_generator(seed=0), n_candidates=5, threshold=0.7)

# --- phase 1: labelled feedback stream (learning on) ---
for q, ctx in question_stream(N_TRAIN, seed=1):
    d = a.answer(q, ctx)
    a.feedback(d.id, d._shadow_text == ctx["truth"])
w_before = dict(a.weights.w)
a.fit_calibration()

# --- phase 2: held-out test (learning off: no feedback calls) ---
rows = []
for q, ctx in question_stream(N_TEST, seed=2):
    d = a.answer(q, ctx)
    rows.append((d, d._shadow_text == ctx["truth"], ctx["slice"]))

raw = [d.raw_confidence for d, _, _ in rows]
cal = [d.confidence for d, _, _ in rows]
y = [int(c) for _, c, _ in rows]
base_acc = sum(y) / len(y)

out = []
P = out.append
P("# Aureon – evaluering på syntetisk model\n")
P(f"Træning: {N_TRAIN} labelede spørgsmål · Test: {N_TEST} (held-out, ingen læring) · 5 kandidater/spørgsmål\n")
P("## 1. Kalibrering (test)\n")
P("| Mål | Rå confidence | Kalibreret |\n|---|---|---|")
P(f"| ECE (lavere er bedre) | {ece(raw, y):.3f} | {ece(cal, y):.3f} |")
P(f"| Brier (lavere er bedre) | {brier(raw, y):.3f} | {brier(cal, y):.3f} |\n")
P("## 2. Risk–coverage (kalibreret confidence, test)\n")
P(f"Baseline (svar altid): accuracy **{base_acc:.3f}**\n")
P("| Tærskel | Coverage | Accuracy på besvarede | Fejl undgået |\n|---|---|---|---|")
base_err = len(y) - sum(y)
for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    ans = [i for i, c in enumerate(cal) if c >= t]
    if not ans:
        P(f"| {t} | 0.000 | – | – |"); continue
    acc = sum(y[i] for i in ans) / len(ans)
    err_left = len(ans) - sum(y[i] for i in ans)
    P(f"| {t} | {len(ans)/len(y):.3f} | {acc:.3f} | {base_err-err_left} af {base_err} |")
P("\n## 3. Blind spots fundet (fra træningsfeedback)\n")
P("| Emne | n | Fejlrate | 95% CI | Sand fejlrate | Flagget |\n|---|---|---|---|---|---|")
for r in a.detector.report():
    P(f"| {r['slice']} | {r['n']} | {r['error_rate']:.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | "
      f"{1-TOPIC_ACC[r['slice']]:.2f}* | {'ja' if r['flagged'] else 'nej'} |")
P("\n*Den sande fejlrate pr. kandidat; systemfejlen er lavere, fordi flertalsafstemning retter noget.\n")
P("## 4. Adaptive vægte\n")
P("| Princip | Start | Efter 1500 labels |\n|---|---|---|")
for k in a.weights.w:
    P(f"| {k} | {1/3:.3f} | {a.weights.w[k]:.3f} |")
P(f"\nKalibrator: a={a.calibrator.a:.2f}, b={a.calibrator.b:.2f}")
print("\n".join(out))
