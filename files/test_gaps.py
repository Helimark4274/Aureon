from aureon import Aureon, Candidate


def _split_gen(q, c, n):
    return [Candidate(str(i)) for i in range(n)]          # total disagreement -> always abstains


def test_fresh_slice_says_not_learned_yet_and_logs_gap():
    a = Aureon(_split_gen)
    d = a.answer("q", {"slice": "tax"})
    assert d.action == "abstain" and d.reason_code == "not_learned_yet"
    assert d.labels_in_slice == 0 and d.labels_needed == 20 and "endnu" in d.message
    assert a.store.gaps() == [("tax", "not_learned_yet", 1)]


def test_after_enough_labels_reason_becomes_low_confidence():
    a = Aureon(_split_gen)
    for _ in range(20):
        d = a.answer("q", {"slice": "tax"})
        a.feedback(d.id, False)
    d = a.answer("q", {"slice": "tax"})
    assert d.reason_code == "low_confidence" and d.labels_needed == 0


def test_answered_has_no_message():
    a = Aureon(lambda q, c, n: [Candidate("Paris")] * n)
    d = a.answer("q")
    assert d.reason_code == "answered" and d.message == ""
