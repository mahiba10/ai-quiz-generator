from utils import quiz_logic as ql
from utils.recommendations import generate_recommendations

def q(i, concept, answer=0, topic="T"):
    return {"id": f"q{i}", "topic": topic, "difficulty": "Easy", "concept": concept,
            "question": f"Q{i}?", "options": ["a", "b", "c", "d"], "answer_index": answer, "explanation": "e"}

BANK = [q(1, "Loops"), q(2, "Loops"), q(3, "Loops"), q(4, "Strings"), q(5, "Strings")]

def test_weak_concepts_come_from_wrong_answers():
    quiz = [BANK[0], BANK[3]]
    result = ql.score_quiz(quiz, {0: 3, 1: 0})   # Loops wrong, Strings right
    rec = generate_recommendations(result, BANK)
    assert [w["concept"] for w in rec["weak_concepts"]] == ["Loops"]

def test_practice_questions_are_unseen_and_same_concept():
    quiz = [BANK[0]]
    rec = generate_recommendations(ql.score_quiz(quiz, {0: 2}), BANK)
    ids = {p["id"] for p in rec["practice_questions"]}
    assert ids == {"q2", "q3"} and "q1" not in ids

def test_perfect_score_has_no_weak_concepts():
    rec = generate_recommendations(ql.score_quiz([BANK[0]], {0: 0}), BANK)
    assert rec["weak_concepts"] == [] and rec["practice_questions"] == []

def test_empty_quiz_feedback():
    rec = generate_recommendations(ql.score_quiz([], {}), BANK)
    assert "No questions" in rec["feedback"]

def test_unanswered_counts_as_weak():
    rec = generate_recommendations(ql.score_quiz([BANK[3]], {}), BANK)
    assert rec["weak_concepts"][0]["concept"] == "Strings"
