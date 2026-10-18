from collections import defaultdict
from aureon.evalkit import is_correct, load_csv, report, run
from aureon.sim import make_generator


def test_is_correct_modes():
    assert is_correct("Svaret er Canberra.", "canberra") and not is_correct("Svaret er Canberra.", "canberra", exact=True)


def test_load_csv_skips_blank_and_defaults_topic(tmp_path):
    f = tmp_path / "q.csv"
    f.write_text("question,truth,topic\nA?,1,x\n,2,y\nB?,3,\n", encoding="utf-8")
    rows = load_csv(str(f))
    assert [r["topic"] for r in rows] == ["x", "default"]


def test_full_run_and_report_on_mock():
    rows = [{"question": f"q{i}", "truth": f"ans-{i}", "topic": "t1" if i % 2 else "t2"} for i in range(300)]
    res = run(rows, make_generator(0, defaultdict(lambda: 0.7)), pass_truth=True)
    md = report(*res)
    assert "Baseline" in md and "Afstå-kurve" in md and len(res[3]) == 150
