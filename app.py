"""AI-Powered Quiz Generator and Personalized Learning Assistant (Streamlit app).
Run with:  streamlit run app.py
Pages: Home -> Start Quiz (setup) -> Quiz -> Results, plus Topic Finder (ML) and History."""
import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ml import train_model
from ml.topic_classifier import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, ModelNotFoundError, TopicClassifier
from utils import quiz_logic as ql
from utils.recommendations import generate_recommendations

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
MAX_QUESTIONS = 20

TOPIC_INFO = {
    "Python Programming": ("🐍", "Syntax, functions, loops and data types."),
    "Machine Learning Basics": ("🤖", "Learning types, evaluation and key terms."),
    "Artificial Intelligence": ("🧠", "Agents, search, NLP and AI ethics."),
    "Data Structures and Algorithms": ("🌳", "Stacks, trees, sorting and complexity."),
    "Computer Networks": ("🌐", "OSI model, protocols and IP addressing."),
    ql.MIXED_TOPIC: ("🎯", "A random mix from every topic."),
}

st.set_page_config(page_title="AI Quiz Generator", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.hero {background: linear-gradient(135deg,#4F46E5,#7C3AED); color:#fff; padding:2rem 1.5rem;
       border-radius:16px; margin-bottom:1.2rem;}
.hero h1 {color:#fff; margin:0 0 .4rem 0; font-size:2rem; line-height:1.25;}
.hero p {margin:0; font-size:1.05rem; opacity:.95;}
.question-text {font-size:1.25rem; font-weight:600; margin:.2rem 0 .8rem 0;}
.small-muted {color:#6B7280; font-size:.9rem;}
.block-container {padding-top:2rem; max-width:860px;}
div[data-testid="stRadio"] label {padding:.25rem 0;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data + state helpers
@st.cache_data
def get_question_bank():
    return ql.load_questions(QUESTIONS_PATH)


@st.cache_resource
def get_classifier():
    return TopicClassifier.load(DEFAULT_MODEL_PATH)


def init_state():
    defaults = {"page": "home", "topic": ql.MIXED_TOPIC, "difficulty": "Easy", "num_questions": 5,
                "quiz": None, "result": None, "recommendations": None, "history": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(page):
    st.session_state.page = page


def set_topic(topic):
    st.session_state.topic = topic


def start_quiz(bank):
    """Create a new quiz in session state (answers live in session state, so reruns don't reset them)."""
    try:
        questions = ql.select_questions(bank, st.session_state.topic, st.session_state.difficulty,
                                        st.session_state.num_questions)
    except ValueError as err:
        st.error(str(err))
        return
    st.session_state.quiz = {"id": uuid.uuid4().hex[:8], "questions": questions, "answers": {}, "current": 0,
                             "submitted": False, "confirm": False, "topic": st.session_state.topic,
                             "difficulty": st.session_state.difficulty}
    st.session_state.result = None
    st.session_state.page = "quiz"
    st.rerun()


def submit_quiz(bank):
    quiz = st.session_state.quiz
    if quiz["submitted"]:       # guard against double submission
        return
    result = ql.score_quiz(quiz["questions"], quiz["answers"])
    quiz["submitted"] = True
    st.session_state.result = result
    st.session_state.recommendations = generate_recommendations(result, bank)
    st.session_state.history.append({
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Topic": quiz["topic"], "Difficulty": quiz["difficulty"],
        "Questions": result["total"], "Correct": result["correct"], "Score (%)": result["percentage"]})
    st.session_state.page = "results"
    st.rerun()


def sidebar():
    st.sidebar.title("🎓 AI Quiz Generator")
    st.sidebar.caption("Supporting SDG 4: Quality Education")
    for label, page in [("🏠 Home", "home"), ("📝 Start a Quiz", "setup"), ("🔎 Topic Finder (ML)", "classifier"),
                        ("📈 History", "history")]:
        st.sidebar.button(label, on_click=go, args=(page,), use_container_width=True, key=f"nav_{page}")
    quiz = st.session_state.quiz
    if quiz and not quiz["submitted"]:
        st.sidebar.button("↩️ Resume current quiz", on_click=go, args=("quiz",), use_container_width=True)


# ----------------------------------------------------------------- pages
def page_home(bank):
    st.markdown("""<div class="hero"><h1>AI-Powered Quiz Generator and Personalized Learning Assistant</h1>
    <p>Practise multiple-choice questions, get instant feedback and find out which topics to revise next.</p></div>""",
                unsafe_allow_html=True)
    st.subheader("How this helps students")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📝 Practise**\n\nChoose a topic, difficulty and number of questions.")
    c2.markdown("**✅ Instant feedback**\n\nSee your score, correct answers and explanations after submitting.")
    c3.markdown("**🎯 Revise smartly**\n\nGet revision suggestions based on the questions you missed.")
    st.write("")
    st.button("Start a Quiz  →", type="primary", on_click=go, args=("setup",), use_container_width=True)
    with st.expander("How does the AI part work?"):
        st.write("A small scikit-learn text classifier (TF-IDF + Logistic Regression/Naive Bayes) predicts which "
                 "topic a learning query belongs to, so it can suggest a quiz. Try it on the **Topic Finder** page. "
                 "Questions come from a curated, human-written question bank, so answers are reliable. "
                 f"The bank currently holds {len(bank)} questions.")


def page_setup(bank):
    st.header("Choose your quiz")
    st.subheader("1. Topic")
    topics = ql.get_topics(bank) + [ql.MIXED_TOPIC]
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        icon, desc = TOPIC_INFO.get(topic, ("📘", ""))
        selected = st.session_state.topic == topic
        with cols[i % 3].container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{topic}**")
            st.caption(desc)
            st.button("✓ Selected" if selected else "Select", key=f"topic_{i}", on_click=set_topic, args=(topic,),
                      type="primary" if selected else "secondary", use_container_width=True)

    st.subheader("2. Difficulty")
    options = ql.DIFFICULTIES + [ql.MIXED_DIFFICULTY]
    st.session_state.difficulty = st.radio("Difficulty", options, horizontal=True, label_visibility="collapsed",
                                           index=options.index(st.session_state.difficulty))

    available = ql.count_available(bank, st.session_state.topic, st.session_state.difficulty)
    st.subheader("3. Number of questions")
    if available == 0:
        st.warning("No questions are available for this combination. Try a different topic or difficulty.")
        return
    max_q = min(MAX_QUESTIONS, available)
    default = min(max(st.session_state.num_questions, 1), max_q)
    st.session_state.num_questions = int(st.number_input(f"Choose 1 to {max_q} questions", min_value=1,
                                                         max_value=max_q, value=default, step=1))
    st.info(f"**{st.session_state.topic}** · {st.session_state.difficulty} · {st.session_state.num_questions} question(s)")
    if st.button("Start Quiz", type="primary", use_container_width=True):
        start_quiz(bank)


def page_quiz(bank):
    quiz = st.session_state.quiz
    if quiz is None:
        st.info("You have no quiz in progress yet.")
        st.button("Choose a quiz", on_click=go, args=("setup",), type="primary")
        return
    if quiz["submitted"]:
        st.info("This quiz has already been submitted.")
        st.button("View results", on_click=go, args=("results",), type="primary")
        return

    questions, answers = quiz["questions"], quiz["answers"]
    total, i = len(questions), quiz["current"]
    q = questions[i]
    st.caption(f"{quiz['topic']} · {quiz['difficulty']}")
    st.progress((i + 1) / total, text=f"Question {i + 1} of {total}  ·  {len(answers)} answered")
    with st.container(border=True):
        st.markdown(f'<div class="question-text">{i + 1}. {q["question"]}</div>', unsafe_allow_html=True)
        # Only the options are shown here; the correct answer is never displayed before submission.
        choice = st.radio("Select one answer", list(range(4)), index=answers.get(i), key=f"ans_{quiz['id']}_{i}",
                          format_func=lambda k: f"{'ABCD'[k]}.  {q['options'][k]}", label_visibility="collapsed")
        if choice is not None:
            answers[i] = choice

    prev_col, next_col = st.columns(2)
    if prev_col.button("← Previous", disabled=i == 0, use_container_width=True):
        quiz["current"] -= 1
        st.rerun()
    if next_col.button("Next →", disabled=i == total - 1, use_container_width=True):
        quiz["current"] += 1
        st.rerun()

    st.divider()
    unanswered = total - len(answers)
    if not quiz["confirm"]:
        if st.button("Submit Quiz", type="primary", use_container_width=True):
            quiz["confirm"] = True
            st.rerun()
    else:
        msg = "Are you sure you want to submit?"
        if unanswered:
            msg += f" You have {unanswered} unanswered question(s); they will count as not correct."
        st.warning(msg)
        yes, no = st.columns(2)
        if yes.button("Yes, submit", type="primary", use_container_width=True):
            submit_quiz(bank)
        if no.button("Keep working", use_container_width=True):
            quiz["confirm"] = False
            st.rerun()


def page_results(bank):
    result, rec, quiz = st.session_state.result, st.session_state.recommendations, st.session_state.quiz
    if result is None:
        st.info("Complete a quiz to see your results here.")
        st.button("Choose a quiz", on_click=go, args=("setup",), type="primary")
        return
    st.header("Your results")
    st.markdown(f"## Score: {result['correct']} / {result['total']} ({result['percentage']}%)")
    st.progress(result["percentage"] / 100)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Questions", result["total"])
    m2.metric("Correct", result["correct"])
    m3.metric("Incorrect", result["incorrect"])
    m4.metric("Unanswered", result["unanswered"])
    st.success(rec["feedback"])

    # Topic-wise performance (needs at least 2 topics); otherwise show concept-wise performance.
    topics_df = pd.DataFrame(result["topic_performance"])
    concepts_df = pd.DataFrame(result["concept_performance"])
    if len(topics_df) >= 2:
        st.subheader("Topic-wise performance")
        st.bar_chart(topics_df.set_index("topic")["percentage"])
    if len(concepts_df) >= 2:
        st.subheader("Concept-wise performance")
        st.dataframe(concepts_df[["topic", "concept", "correct", "total", "percentage"]], hide_index=True,
                     use_container_width=True)
    else:
        st.caption("Not enough questions to show a meaningful performance breakdown.")

    wrong = [d for d in result["details"] if not d["is_correct"]]
    st.subheader(f"Questions to review ({len(wrong)})")
    if not wrong:
        st.write("🎉 You answered every question correctly.")
    for d in wrong:
        with st.expander(f"Q{d['position']}. {d['question']}"):
            mine = f"{'ABCD'[d['selected_index']]}. {d['options'][d['selected_index']]}" if d["answered"] else "Not answered"
            st.markdown(f"**Your answer:** {mine}")
            st.markdown(f"**Correct answer:** {'ABCD'[d['correct_index']]}. {d['options'][d['correct_index']]}")
            st.markdown(f"**Explanation:** {d['explanation']}")

    st.subheader("All answers and explanations")
    for d in result["details"]:
        with st.expander(f"{'✅' if d['is_correct'] else '❌'} Q{d['position']}. {d['question']}"):
            for k, text in enumerate(d["options"]):
                tag = " ← correct" if k == d["correct_index"] else (" ← your answer" if k == d["selected_index"] else "")
                st.write(f"{'ABCD'[k]}. {text}{tag}")
            st.caption(d["explanation"])

    st.subheader("Personalised recommendations")
    if rec["weak_concepts"]:
        for w in rec["weak_concepts"]:
            st.markdown(f"- 📌 {w['tip']}")
    else:
        st.write("No weak concepts found in this quiz. Try a harder difficulty next.")
    if rec["practice_questions"]:
        st.markdown("**Try these related practice questions** (not used in this quiz):")
        for p in rec["practice_questions"]:
            with st.expander(f"{p['topic']} · {p['concept']} · {p['difficulty']}: {p['question']}"):
                for k, text in enumerate(p["options"]):
                    st.write(f"{'ABCD'[k]}. {text}")
                with st.expander("Show answer"):
                    st.write(f"**{p['options'][p['answer_index']]}** — {p['explanation']}")

    a, b = st.columns(2)
    if a.button("Retake with same settings", use_container_width=True):
        st.session_state.topic, st.session_state.difficulty = quiz["topic"], quiz["difficulty"]
        st.session_state.num_questions = result["total"]
        start_quiz(bank)
    b.button("New quiz", on_click=go, args=("setup",), type="primary", use_container_width=True)


def page_classifier(bank):
    st.header("🔎 Topic Finder (machine learning)")
    tab_try, tab_eval = st.tabs(["Try the classifier", "Model evaluation"])
    with tab_try:
        st.write("Type what you want to learn. A TF-IDF + scikit-learn classifier predicts the most likely quiz topic.")
        try:
            clf = get_classifier()
        except ModelNotFoundError as err:
            st.warning(f"The model has not been trained yet. {err}")
            if st.button("Train the model now"):
                with st.spinner("Training..."):
                    train_model.run()
                st.cache_resource.clear()
                st.rerun()
            clf = None
        if clf:
            text = st.text_area("What do you want to learn?", max_chars=500,
                                placeholder="e.g. I want to learn about functions and loops in Python.")
            if st.button("Predict topic", type="primary"):
                r = clf.predict(text)
                if r["status"] == "empty":
                    st.info("Please type a sentence first.")
                elif r["status"] in ("unknown", "invalid"):
                    st.warning("I don't recognise any of those words from my training data, so I can't predict a topic. "
                               "Try mentioning a concept such as 'loops', 'overfitting', 'stack' or 'TCP'.")
                else:
                    pct = round(r["confidence"] * 100)
                    if r["status"] == "ok":
                        st.success(f"Predicted topic: **{r['topic']}** (confidence {pct}%)")
                    else:
                        st.warning(f"I'm not sure. My best guess is **{r['topic']}** ({pct}% confidence), "
                                   "which is low. Try adding more detail.")
                    st.bar_chart(pd.Series(r["probabilities"], name="probability"))
                    st.button(f"Start a {r['topic']} quiz", on_click=lambda t=r["topic"]: (set_topic(t), go("setup")))
            st.caption("Confidence values are the model's own estimates from a small dataset and can be wrong.")
    with tab_eval:
        if not DEFAULT_METRICS_PATH.exists():
            st.info("No evaluation results yet. Run `python ml/train_model.py` to measure them.")
            return
        m = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"Model: **{m['chosen_model']}** · random seed {m['random_state']} · "
                 f"{m['n_train']} training / {m['n_test']} test sentences (stratified {int((1 - m['test_size']) * 100)}/"
                 f"{int(m['test_size'] * 100)} split).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['test_accuracy']:.2f}")
        c2.metric("Macro precision", f"{m['test_macro_precision']:.2f}")
        c3.metric("Macro recall", f"{m['test_macro_recall']:.2f}")
        c4.metric("Macro F1", f"{m['test_macro_f1']:.2f}")
        report = pd.DataFrame(m["classification_report"]).T.loc[m["labels"], ["precision", "recall", "f1-score", "support"]]
        st.dataframe(report.round(2), use_container_width=True)
        st.write("Confusion matrix (rows = true topic, columns = predicted topic):")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]), use_container_width=True)
        st.warning("Honest note: the test set is very small, so these numbers can change a lot with a different split "
                   "or more data. They show the pipeline works, not that it is production-ready.")


def page_history():
    st.header("📈 Quiz history")
    history = st.session_state.history
    if not history:
        st.info("No quizzes completed yet. Finish a quiz and it will appear here.")
        st.button("Start a Quiz", on_click=go, args=("setup",), type="primary")
        return
    df = pd.DataFrame(history)
    st.dataframe(df, hide_index=True, use_container_width=True)
    if len(df) >= 2:
        st.line_chart(df["Score (%)"])
    st.caption("History is kept only for this browser session; it clears when you close or refresh the page.")
    st.download_button("Download history (CSV)", df.to_csv(index=False), "quiz_history.csv", "text/csv")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


def main():
    init_state()
    try:
        bank = get_question_bank()
    except (FileNotFoundError, ValueError) as err:
        st.error(f"The question bank could not be loaded: {err}")
        st.stop()
    if not bank:
        st.error("The question bank contains no valid questions.")
        st.stop()
    sidebar()
    page = st.session_state.page
    if page == "home":
        page_home(bank)
    elif page == "setup":
        page_setup(bank)
    elif page == "quiz":
        page_quiz(bank)
    elif page == "results":
        page_results(bank)
    elif page == "classifier":
        page_classifier(bank)
    elif page == "history":
        page_history()


main()
