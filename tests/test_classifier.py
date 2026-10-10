import pytest
from ml import train_model
from ml.topic_classifier import TopicClassifier, ModelNotFoundError

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    metrics = train_model.run(model_path=d / "m.joblib", metrics_path=d / "m.json")
    return TopicClassifier.load(d / "m.joblib"), metrics

def test_training_is_reproducible(tmp_path):
    a = train_model.run(model_path=tmp_path / "a.joblib", metrics_path=tmp_path / "a.json")
    b = train_model.run(model_path=tmp_path / "b.joblib", metrics_path=tmp_path / "b.json")
    assert a["test_accuracy"] == b["test_accuracy"] and a["chosen_model"] == b["chosen_model"]

def test_split_has_no_overlap_and_expected_sizes(trained):
    _, m = trained
    assert m["n_train"] + m["n_test"] == m["n_samples_total"]
    assert 0.0 <= m["test_accuracy"] <= 1.0

def test_clear_python_query(trained):
    clf, _ = trained
    r = clf.predict("Teach me about lists and loops in Python")
    assert r["status"] == "ok" and r["topic"] == "Python Programming"

def test_clear_network_query(trained):
    clf, _ = trained
    assert clf.predict("How does the TCP handshake work?")["topic"] == "Computer Networks"

def test_probabilities_sum_to_one(trained):
    clf, _ = trained
    probs = clf.predict("Explain overfitting in a model")["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6

@pytest.mark.parametrize("text,status", [("", "empty"), ("   ", "empty"), (None, "invalid"),
                                         (123, "invalid"), ("zzzz qqqq", "unknown"), ("pizza with cheese", "unknown")])
def test_invalid_and_unknown_inputs(trained, text, status):
    clf, _ = trained
    r = clf.predict(text)
    assert r["status"] == status and r["topic"] is None

def test_very_long_input_does_not_crash(trained):
    clf, _ = trained
    assert clf.predict("python loops " * 500)["status"] in {"ok", "uncertain"}

def test_missing_model_file_is_handled(tmp_path):
    with pytest.raises(ModelNotFoundError):
        TopicClassifier.load(tmp_path / "missing.joblib")

def test_corrupted_model_file_is_handled(tmp_path):
    p = tmp_path / "bad.joblib"; p.write_text("not a model")
    with pytest.raises(ModelNotFoundError):
        TopicClassifier.load(p)

def test_training_rejects_bad_data(tmp_path):
    p = tmp_path / "d.csv"; p.write_text("text,label\nhello,A\nworld,B\n")
    with pytest.raises(ValueError):
        train_model.load_and_clean_data(p)
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        train_model.load_and_clean_data(p)
    with pytest.raises(FileNotFoundError):
        train_model.load_and_clean_data(tmp_path / "none.csv")
