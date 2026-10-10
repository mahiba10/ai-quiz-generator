"""End-to-end smoke test of the Streamlit app using Streamlit's built-in AppTest."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

def button(at, label):
    return next(b for b in at.button if label in b.label)

def test_full_quiz_flow():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    button(at, "Start a Quiz").click().run()          # home -> setup
    button(at, "Start Quiz").click().run()            # setup -> quiz (default 5 questions)
    assert not at.exception and at.session_state.page == "quiz"
    quiz = at.session_state.quiz
    total = len(quiz["questions"])
    # answer the first question correctly, then go through the rest choosing option 0
    for i in range(total):
        correct = quiz["questions"][i]["answer_index"]
        at.radio[0].set_value(correct if i == 0 else 0).run()
        if i < total - 1:
            button(at, "Next").click().run()
    # answers survive navigation
    assert len(at.session_state.quiz["answers"]) == total
    button(at, "Submit Quiz").click().run()
    button(at, "Yes, submit").click().run()
    assert not at.exception and at.session_state.page == "results"
    result = at.session_state.result
    expected = sum(1 for i, q in enumerate(quiz["questions"]) if at.session_state.quiz["answers"][i] == q["answer_index"])
    assert result["correct"] == expected and result["total"] == total
    assert len(at.session_state.history) == 1

def test_classifier_page_empty_input():
    at = AppTest.from_file(APP, default_timeout=30).run()
    button(at, "Topic Finder").click().run()
    assert not at.exception
    button(at, "Predict topic").click().run()
    assert any("type a sentence" in i.value for i in at.info)
