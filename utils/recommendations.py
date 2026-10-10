"""Personalised recommendations built ONLY from the student's actual answers and question metadata
(topic, concept, difficulty). This is rule-based logic, not machine learning."""
from utils.quiz_logic import calculate_percentage


def feedback_message(result):
    """Encouraging feedback that describes this quiz only (no claims about ability)."""
    if result["total"] == 0:
        return "No questions were answered in this quiz."
    pct = result["percentage"]
    if pct >= 80:
        return f"Great work - you answered {pct}% of this quiz correctly. Keep practising to stay sharp."
    if pct >= 50:
        return f"Good effort - {pct}% correct on this quiz. Reviewing the missed concepts below should help you improve."
    return f"This quiz scored {pct}%. That is a useful starting point: revise the concepts below and try again."


def generate_recommendations(result, question_bank, max_practice=3):
    """Return feedback, weak concepts and unseen practice questions for those concepts."""
    weak = [c for c in result["concept_performance"] if c["missed"] > 0]
    weak.sort(key=lambda c: (-c["missed"], c["percentage"], c["concept"]))
    weak_concepts = [{"topic": c["topic"], "concept": c["concept"], "missed": c["missed"], "total": c["total"],
                      "tip": f"Revise '{c['concept']}' in {c['topic']}: re-read the explanations for the "
                             f"{c['missed']} question(s) you missed."} for c in weak]

    used_ids = {d["id"] for d in result["details"]}
    practice = []
    for w in weak_concepts:
        extra = [q for q in question_bank
                 if q["topic"] == w["topic"] and q["concept"] == w["concept"] and q["id"] not in used_ids]
        for q in extra:
            if len(practice) < max_practice and q["id"] not in {p["id"] for p in practice}:
                practice.append(q)
    return {"feedback": feedback_message(result), "weak_concepts": weak_concepts, "practice_questions": practice}
