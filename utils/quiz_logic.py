"""Quiz logic: loading, validating, selecting and scoring questions.
There is NO machine learning in this file - scoring is plain arithmetic on the student's answers."""
import json
import random
from pathlib import Path

DIFFICULTIES = ["Easy", "Medium", "Hard"]
MIXED_TOPIC = "Mixed (All Topics)"
MIXED_DIFFICULTY = "Mixed"
REQUIRED_FIELDS = ("id", "topic", "difficulty", "concept", "question", "options", "answer_index", "explanation")


def is_valid_question(item):
    """True only if a question has every field, exactly 4 distinct non-empty options and a valid answer."""
    if not isinstance(item, dict) or any(f not in item for f in REQUIRED_FIELDS):
        return False
    for f in ("id", "topic", "concept", "question", "explanation"):
        if not isinstance(item[f], str) or not item[f].strip():
            return False
    if item["difficulty"] not in DIFFICULTIES:
        return False
    opts = item["options"]
    if not isinstance(opts, list) or len(opts) != 4:
        return False
    if any(not isinstance(o, str) or not o.strip() for o in opts) or len(set(opts)) != 4:
        return False
    idx = item["answer_index"]
    return isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < 4


def load_questions(path):
    """Load the question bank. Malformed entries and duplicate ids are skipped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Question file is not valid JSON: {err}") from err
    if not isinstance(raw, list):
        raise ValueError("Question file must contain a list of questions.")
    valid, seen = [], set()
    for item in raw:
        if is_valid_question(item) and item["id"] not in seen:
            seen.add(item["id"])
            valid.append(item)
    return valid


def get_topics(questions):
    return sorted({q["topic"] for q in questions})


def filter_questions(questions, topic, difficulty):
    """Return questions matching the topic/difficulty (Mixed = no filter)."""
    result = questions
    if topic != MIXED_TOPIC:
        result = [q for q in result if q["topic"] == topic]
    if difficulty != MIXED_DIFFICULTY:
        result = [q for q in result if q["difficulty"] == difficulty]
    return list(result)


def count_available(questions, topic, difficulty):
    """Number of distinct questions available for a selection."""
    return len({q["question"].strip().lower() for q in filter_questions(questions, topic, difficulty)})


def select_questions(questions, topic, difficulty, num_questions, seed=None):
    """Pick `num_questions` random, non-duplicate questions and shuffle each question's options.
    The returned copies keep `answer_index` pointing at the correct option after shuffling."""
    if isinstance(num_questions, bool) or not isinstance(num_questions, int) or num_questions < 1:
        raise ValueError("Number of questions must be a whole number of at least 1.")
    pool, seen = [], set()
    for q in filter_questions(questions, topic, difficulty):
        key = q["question"].strip().lower()
        if key not in seen:      # avoid duplicate questions in one quiz
            seen.add(key)
            pool.append(q)
    if not pool:
        raise ValueError("No questions are available for this topic and difficulty.")
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(num_questions, len(pool)))
    quiz = []
    for q in chosen:
        correct_text = q["options"][q["answer_index"]]
        options = q["options"][:]
        rng.shuffle(options)
        copy = dict(q)
        copy["options"] = options
        copy["answer_index"] = options.index(correct_text)
        quiz.append(copy)
    return quiz


def check_answer(question, selected_index):
    """True if the selected option index is correct. None (unanswered) is incorrect."""
    if selected_index is None:
        return False
    if isinstance(selected_index, bool) or not isinstance(selected_index, int) or not 0 <= selected_index < len(question["options"]):
        raise ValueError(f"Invalid answer choice: {selected_index!r}")
    return selected_index == question["answer_index"]


def calculate_percentage(correct, total):
    """Percentage rounded to 1 decimal. An empty quiz (total 0) gives 0.0."""
    for value in (correct, total):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Correct and total must be non-negative whole numbers.")
    if correct > total:
        raise ValueError("Correct answers cannot exceed total questions.")
    return 0.0 if total == 0 else round(correct / total * 100, 1)


def _group_performance(details, key_fn, label_names):
    groups = {}
    for d in details:
        g = groups.setdefault(key_fn(d), {"total": 0, "correct": 0})
        g["total"] += 1
        g["correct"] += int(d["is_correct"])
    rows = []
    for key, g in sorted(groups.items()):
        row = dict(zip(label_names, key if isinstance(key, tuple) else (key,)))
        row.update(total=g["total"], correct=g["correct"], missed=g["total"] - g["correct"],
                   percentage=calculate_percentage(g["correct"], g["total"]))
        rows.append(row)
    return rows


def score_quiz(questions, answers):
    """Score a quiz from the student's actual answers.
    `answers` maps question position (0-based) to the chosen option index (or missing = unanswered)."""
    details = []
    for i, q in enumerate(questions):
        selected = answers.get(i)
        is_correct = check_answer(q, selected)
        details.append({
            "position": i + 1, "id": q["id"], "topic": q["topic"], "difficulty": q["difficulty"],
            "concept": q["concept"], "question": q["question"], "options": q["options"],
            "selected_index": selected, "correct_index": q["answer_index"],
            "answered": selected is not None, "is_correct": is_correct, "explanation": q["explanation"]})
    total = len(details)
    correct = sum(d["is_correct"] for d in details)
    unanswered = sum(not d["answered"] for d in details)
    return {
        "total": total, "correct": correct,
        "incorrect": total - correct - unanswered,   # answered but wrong
        "unanswered": unanswered,
        "percentage": calculate_percentage(correct, total),
        "details": details,
        "topic_performance": _group_performance(details, lambda d: d["topic"], ["topic"]),
        "concept_performance": _group_performance(details, lambda d: (d["topic"], d["concept"]), ["topic", "concept"]),
    }
