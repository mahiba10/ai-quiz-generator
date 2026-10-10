import json
import pytest
from utils import quiz_logic as ql

def make_q(i, topic="Python Programming", diff="Easy", answer=1, concept="Basics"):
    return {"id": f"t-{i}", "topic": topic, "difficulty": diff, "concept": concept,
            "question": f"Question {i}?", "options": ["A", "B", "C", "D"],
            "answer_index": answer, "explanation": "Because."}

QUESTIONS = [make_q(i) for i in range(6)]

# ---------- answer checking ----------
def test_check_answer_correct_and_wrong():
    q = make_q(1, answer=2)
    assert ql.check_answer(q, 2) is True
    assert ql.check_answer(q, 0) is False

def test_check_answer_unanswered_is_incorrect():
    assert ql.check_answer(make_q(1), None) is False

@pytest.mark.parametrize("bad", [4, -1, "1", 1.5, True])
def test_check_answer_invalid_input(bad):
    with pytest.raises(ValueError):
        ql.check_answer(make_q(1), bad)

# ---------- percentage ----------
def test_percentage_basic():
    assert ql.calculate_percentage(8, 10) == 80.0
    assert ql.calculate_percentage(1, 3) == 33.3

def test_percentage_empty_quiz_is_zero():
    assert ql.calculate_percentage(0, 0) == 0.0

@pytest.mark.parametrize("c,t", [(-1, 5), (6, 5), (1.5, 5), ("1", 5)])
def test_percentage_invalid(c, t):
    with pytest.raises(ValueError):
        ql.calculate_percentage(c, t)

# ---------- scoring ----------
def test_score_calculation_from_actual_answers():
    qs = [make_q(i, answer=1) for i in range(10)]
    answers = {i: 1 for i in range(8)}      # 8 correct
    answers[8] = 0                          # 1 wrong; question 10 left unanswered
    r = ql.score_quiz(qs, answers)
    assert (r["total"], r["correct"], r["incorrect"], r["unanswered"]) == (10, 8, 1, 1)
    assert r["percentage"] == 80.0
    assert r["correct"] + r["incorrect"] + r["unanswered"] == r["total"]

def test_score_all_wrong_and_all_right():
    qs = [make_q(i, answer=0) for i in range(4)]
    assert ql.score_quiz(qs, {i: 0 for i in range(4)})["percentage"] == 100.0
    assert ql.score_quiz(qs, {i: 3 for i in range(4)})["percentage"] == 0.0

def test_empty_quiz_handling():
    r = ql.score_quiz([], {})
    assert r["total"] == 0 and r["percentage"] == 0.0 and r["details"] == []

def test_topic_and_concept_breakdown():
    qs = [make_q(0, topic="A", concept="x", answer=0), make_q(1, topic="B", concept="y", answer=0)]
    r = ql.score_quiz(qs, {0: 0, 1: 2})
    by_topic = {row["topic"]: row for row in r["topic_performance"]}
    assert by_topic["A"]["percentage"] == 100.0 and by_topic["B"]["percentage"] == 0.0

# ---------- selection ----------
def test_selection_no_duplicates_and_count():
    quiz = ql.select_questions(QUESTIONS, "Python Programming", "Easy", 4, seed=1)
    assert len(quiz) == 4 and len({q["id"] for q in quiz}) == 4

def test_selection_removes_duplicate_question_text():
    dup = [make_q(1), make_q(2)]
    dup[1]["question"] = dup[0]["question"]
    assert len(ql.select_questions(dup, "Python Programming", "Easy", 2)) == 1

def test_selection_is_clamped_to_available():
    assert len(ql.select_questions(QUESTIONS, "Python Programming", "Easy", 50)) == 6

def test_shuffling_keeps_correct_answer_correct():
    for seed in range(20):
        quiz = ql.select_questions([make_q(1, answer=1)], "Python Programming", "Easy", 1, seed=seed)
        assert quiz[0]["options"][quiz[0]["answer_index"]] == "B"

def test_selection_does_not_modify_original_bank():
    before = json.dumps(QUESTIONS)
    ql.select_questions(QUESTIONS, "Python Programming", "Easy", 3, seed=3)
    assert json.dumps(QUESTIONS) == before

def test_selection_same_seed_same_result():
    a = ql.select_questions(QUESTIONS, "Python Programming", "Easy", 3, seed=7)
    b = ql.select_questions(QUESTIONS, "Python Programming", "Easy", 3, seed=7)
    assert a == b

@pytest.mark.parametrize("n", [0, -3, "5", 2.5, None])
def test_selection_invalid_number(n):
    with pytest.raises(ValueError):
        ql.select_questions(QUESTIONS, "Python Programming", "Easy", n)

def test_selection_no_matching_questions():
    with pytest.raises(ValueError):
        ql.select_questions(QUESTIONS, "Computer Networks", "Hard", 3)

def test_mixed_selection_uses_all_topics():
    qs = [make_q(0, topic="A"), make_q(1, topic="B", diff="Hard")]
    assert len(ql.select_questions(qs, ql.MIXED_TOPIC, ql.MIXED_DIFFICULTY, 5)) == 2

# ---------- loading / malformed data ----------
def test_load_real_question_bank_is_valid():
    from pathlib import Path
    qs = ql.load_questions(Path(__file__).resolve().parent.parent / "data" / "questions.json")
    assert len(qs) >= 60
    assert len({q["id"] for q in qs}) == len(qs)
    for topic in ql.get_topics(qs):
        for d in ql.DIFFICULTIES:
            assert ql.count_available(qs, topic, d) >= 5

def test_load_skips_malformed_entries(tmp_path):
    bad = [make_q(1), {"id": "x"}, {**make_q(2), "answer_index": 9}, {**make_q(3), "options": ["A", "A", "B", "C"]}]
    p = tmp_path / "q.json"; p.write_text(json.dumps(bad))
    assert [q["id"] for q in ql.load_questions(p)] == ["t-1"]

def test_load_missing_and_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ql.load_questions(tmp_path / "nope.json")
    p = tmp_path / "bad.json"; p.write_text("{not json")
    with pytest.raises(ValueError):
        ql.load_questions(p)
    p.write_text('{"a": 1}')
    with pytest.raises(ValueError):
        ql.load_questions(p)
