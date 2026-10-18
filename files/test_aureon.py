import random
from aureon import Aureon, Candidate, PlattCalibrator, MetaPatternDetector, AdaptiveWeights, ece
from aureon.meta import wilson_interval
from aureon.sim import make_generator, question_stream


def test_calibrator_reduces_ece_on_overconfident_data():
    rng = random.Random(0)
    conf = [rng.uniform(0.7, 1.0) for _ in range(3000)]
    y = [1 if rng.random() < (c - 0.3) else 0 for c in conf]   # true acc ≈ conf-0.3 -> overconfident
    cal = PlattCalibrator().fit(conf[:2000], y[:2000])
    assert ece(conf[2000:], y[2000:]) > 0.2
    assert ece([cal(c) for c in conf[2000:]], y[2000:]) < 0.08


def test_calibrator_identity_when_too_little_data():
    c = PlattCalibrator().fit([0.9] * 5, [1] * 5)
    assert not c.fitted and c(0.8) == 0.8


def test_wilson_bounds():
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_detector_flags_bad_slice_only():
    d = MetaPatternDetector(min_n=20)
    rng = random.Random(1)
    for _ in range(200):
        d.record("good", rng.random() < 0.95)
        d.record("bad", rng.random() < 0.40)
    assert d.is_blind_spot("bad") and not d.is_blind_spot("good")
    assert not d.is_blind_spot("unseen")


def test_weights_shift_to_informative_principle_and_stay_normalised():
    w = AdaptiveWeights(["good", "noise"])
    rng = random.Random(2)
    for _ in range(300):
        correct = rng.random() < 0.5
        w.update({"good": 0.9 if correct else 0.1, "noise": 0.5}, correct)
    assert w.w["good"] > w.w["noise"] and abs(sum(w.w.values()) - 1) < 1e-9


def test_constant_optimist_principle_does_not_take_over_at_high_base_rate():
    w = AdaptiveWeights(["disc", "always_one"])
    rng = random.Random(3)
    for _ in range(1500):
        correct = rng.random() < 0.85
        w.update({"disc": (0.9 if correct else 0.2) + rng.uniform(-0.05, 0.05), "always_one": 1.0}, correct)
    assert w.w["disc"] > 0.7
    assert min(w.w.values()) > 0


def test_engine_abstains_on_disagreement_and_answers_on_agreement():
    a = Aureon(lambda q, c, n: [Candidate(str(i)) for i in range(n)], threshold=0.7)
    assert a.answer("x").action == "abstain"
    b = Aureon(lambda q, c, n: [Candidate("Paris")] * n, threshold=0.7)
    d = b.answer("x")
    assert d.action == "answer" and d.text == "Paris" and d.agreement == 1.0


def test_engine_handles_no_candidates():
    assert Aureon(lambda q, c, n: []).answer("x").action == "abstain"


def test_grounding_principle_uses_sources():
    gen = lambda q, c, n: [Candidate("the moon orbits earth")] * n
    a = Aureon(gen, threshold=0.0)
    hi = a.answer("q", {"sources": ["The moon orbits earth every month"]}).principle_scores["grounding"]
    lo = a.answer("q", {"sources": ["bananas are yellow"]}).principle_scores["grounding"]
    assert hi > 0.9 and lo < 0.1


def test_selective_accuracy_beats_baseline_on_held_out():
    a = Aureon(make_generator(0), threshold=0.7)
    for q, ctx in question_stream(1500, 1):
        d = a.answer(q, ctx); a.feedback(d.id, d._shadow_text == ctx["truth"])
    a.fit_calibration()
    res = []
    for q, ctx in question_stream(2000, 2):
        d = a.answer(q, ctx); res.append((d, d._shadow_text == ctx["truth"]))
    base = sum(c for _, c in res) / len(res)
    answered = [c for d, c in res if d.action == "answer"]
    assert len(answered) > 0.3 * len(res)                      # still useful coverage
    assert sum(answered) / len(answered) > base + 0.05         # and meaningfully more accurate
    assert any(r["flagged"] and r["slice"] in ("medicine", "law") for r in a.detector.report())
