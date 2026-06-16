import logging
import re
from pathlib import Path

import joblib


logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"

FALLBACK_KEYWORDS = {
    "bug": ["error", "undefined", "crash", "failed", "cannot", "exception", "bug", "broken"],
    "feature": ["feature", "request", "support", "add", "enhance", "proposal"],
    "question": ["how", "why", "question", "help", "can i", "is it possible"],
    "documentation": ["docs", "documentation", "guide", "example", "readme"],
}


class IssueCategoryPredictor:
    def __init__(self, model_path: Path = MODEL_DIR / "logisticregression.pkl") -> None:
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning("Prediction model not found: %s", self.model_path)
            return

        self.model = joblib.load(self.model_path)
        logger.info("Loaded prediction model: %s", self.model_path)

    def predict(self, text: str) -> tuple[str, float]:
        normalized = normalize_text(text)
        if self.model is None:
            return fallback_predict(normalized)

        probabilities = self.model.predict_proba([normalized])[0]
        class_index = int(probabilities.argmax())
        category = str(self.model.classes_[class_index])
        confidence = float(probabilities[class_index])
        return category, confidence


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text)
    text = re.sub(r"[^a-z0-9\s#+._-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fallback_predict(text: str) -> tuple[str, float]:
    scores = {
        category: sum(keyword in text for keyword in keywords)
        for category, keywords in FALLBACK_KEYWORDS.items()
    }
    category = max(scores, key=scores.get)
    score = scores[category]
    if score == 0:
        return "bug", 0.55
    return category, min(0.65 + (score * 0.1), 0.9)
