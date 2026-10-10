"""Load the trained model and predict the topic of a short text.
This file is independent of the quiz scoring logic."""
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "topic_classifier.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "models" / "metrics.json"

CONFIDENCE_THRESHOLD = 0.40  # below this we say "not sure" instead of guessing
MAX_INPUT_CHARS = 500


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist yet."""


class TopicClassifier:
    def __init__(self, pipeline, threshold=CONFIDENCE_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found at {path}. Run `python ml/train_model.py` to create it.")
        try:
            return cls(joblib.load(path))
        except Exception as err:  # corrupted file or incompatible scikit-learn version
            raise ModelNotFoundError(f"Could not load the model ({err}). Please retrain it.") from err

    def predict(self, text):
        """Return a dict: status ('ok' | 'uncertain' | 'unknown' | 'empty' | 'invalid'),
        topic, confidence and per-topic probabilities."""
        result = {"status": "invalid", "topic": None, "confidence": 0.0, "probabilities": {}}
        if not isinstance(text, str):
            return result
        text = text.strip()[:MAX_INPUT_CHARS]
        if not text:
            result["status"] = "empty"
            return result

        # If none of the words were seen in training, the model has no evidence at all.
        features = self.pipeline.named_steps["tfidf"].transform([text])
        if features.nnz == 0:
            result["status"] = "unknown"
            return result

        probs = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        ranked = dict(sorted(zip(classes, (float(p) for p in probs)), key=lambda kv: -kv[1]))
        top_topic, top_prob = next(iter(ranked.items()))
        result.update(topic=top_topic, confidence=top_prob, probabilities=ranked,
                      status="ok" if top_prob >= self.threshold else "uncertain")
        return result
