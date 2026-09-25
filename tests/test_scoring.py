import yaml

from evals.scoring import check_citation, keyword_check, score_answer, summarize
from evals.run_eval import EVAL_DIR


def _repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("\n".join(f"line{i}" for i in range(10)))
    return tmp_path


def test_hallucination_check(tmp_path):
    root = _repo(tmp_path)
    assert check_citation("app/x.py", [1, 10], root) is None
    assert check_citation("./app/x.py", [3, 4], root) is None
    assert "does not exist" in check_citation("app/nope.py", [1, 2], root)
    assert "does not exist" in check_citation("../../etc/passwd", [1, 2], root)
    assert "invalid range" in check_citation("app/x.py", [5, 11], root)   # past EOF
    assert "invalid range" in check_citation("app/x.py", [6, 5], root)    # reversed
    assert "invalid range" in check_citation("app/x.py", [0, 3], root)    # 0-indexed


def test_keywords_whole_word_with_alternatives():
    assert keyword_check("Returns 401 Unauthorized", ["401|unauthorized"])["ok"]
    assert keyword_check("uses POST /seen", ["/seen"])["ok"]
    assert keyword_check("threshold is 3.0", ["3.0|3"])["ok"]
    assert not keyword_check("the note says so", ["no"])["ok"]      # "no" inside "note" doesn't count
    assert keyword_check("uses price_move and volume", ["price"])["ok"]  # identifiers count
    assert keyword_check("x", ["a", "x"])["missing"] == ["a"]


def test_score_and_summary(tmp_path):
    root = _repo(tmp_path)
    q = {"expected_files": ["app/x.py"], "must_mention": ["line"]}
    good = {"answer": "see line 3", "confidence": "high", "cited_files": [{"path": "app/x.py", "lines": [1, 3]}]}
    bad = {"answer": "somewhere", "confidence": "high", "cited_files": [{"path": "app/fake.py", "lines": [1, 3]}]}
    rows = [{"answer": a, **score_answer(q, a, root)} for a in (good, bad)]
    assert rows[0]["strict_correct"] and not rows[1]["strict_correct"]
    s = summarize(rows, use_judge=False)
    assert s["strict_correct"] == 1 and s["hallucinated_citations"] == 1
    assert s["calibration"]["high"] == {"n": 2, "correct": 1}


def test_negative_question_skips_citation_check(tmp_path):
    q = {"expected_files": [], "must_mention": ["no"]}
    a = {"answer": "No, there is none.", "confidence": "high", "cited_files": []}
    r = score_answer(q, a, _repo(tmp_path))
    assert r["citation_hit"] is None and r["strict_correct"]


def test_questions_file_is_well_formed():
    spec = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text())
    ids = [q["id"] for q in spec["questions"]]
    assert len(ids) == len(set(ids)) and 15 <= len(ids) <= 20
    for q in spec["questions"]:
        assert {"question", "type", "expected_files", "must_mention", "reference_answer"} <= q.keys()
