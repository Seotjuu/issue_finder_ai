import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, silhouette_score, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer


try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
except ImportError:  # pragma: no cover - handled at runtime for lightweight environments.
    EarlyStopping = None
    Dense = None
    Embedding = None
    GRU = None
    LSTM = None
    Sequential = None
    pad_sequences = None
    Tokenizer = None
    to_categorical = None


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "issues.json"
MODEL_DIR = BASE_DIR / "saved_models"
FRONTEND_LIB_DIR = BASE_DIR.parent / "frontend" / "lib"
RANDOM_STATE = 42
TARGET_LABELS = ["bug", "feature", "question", "documentation"]
LABEL_PRIORITY = ["bug", "question", "documentation", "feature"]
CLUSTER_COLORS = ["#e11d48", "#16a34a", "#2563eb", "#facc15"]
CLUSTER_SAMPLE_PER_LABEL = 100
CLUSTER_COUNT = 4
CLUSTER_STOP_WORDS = {
    "affected",
    "agree",
    "apply",
    "automated",
    "behavior",
    "browser",
    "browsers",
    "code",
    "codesandbox",
    "conduct",
    "environment",
    "expected",
    "getting",
    "follow",
    "happens",
    "link",
    "log",
    "main",
    "no",
    "output",
    "problem",
    "project",
    "provided",
    "related",
    "relevant",
    "reproduce",
    "reproduces",
    "response",
    "sandbox",
    "seeing",
    "select",
    "stack",
    "stackblitz",
    "steps",
    "version",
    "versions",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_issues(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run collect_issues.py first or create backend/dataset/issues.json."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("issues.json must contain a JSON array.")

    return pd.DataFrame(data)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text)
    text = re.sub(r"[^a-z0-9\s#+._-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label(label: str) -> str | None:
    normalized = label.lower().strip()

    if any(keyword in normalized for keyword in ["bug", "error", "crash", "regression", "defect"]):
        return "bug"
    if any(keyword in normalized for keyword in ["feature", "enhancement", "proposal", "request"]):
        return "feature"
    if any(keyword in normalized for keyword in ["question", "help", "discussion", "support"]):
        return "question"
    if any(keyword in normalized for keyword in ["documentation", "docs", "doc", "guide", "example"]):
        return "documentation"

    return None


def choose_target_label(labels: list[Any]) -> str | None:
    mapped = {mapped for label in labels if isinstance(label, str) and (mapped := map_label(label))}
    for label in LABEL_PRIORITY:
        if label in mapped:
            return label
    return None


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"title", "body", "labels"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    prepared = df.copy()
    prepared["title"] = prepared["title"].fillna("").astype(str)
    prepared["body"] = prepared["body"].fillna("").astype(str)
    prepared = prepared[prepared["title"].str.strip().ne("")]
    if "labels_normalized" in prepared.columns:
        prepared["target"] = prepared["labels_normalized"].apply(
            lambda labels: labels[0] if isinstance(labels, list) and labels else None
        )
    else:
        prepared["target"] = prepared["labels"].apply(
            lambda labels: choose_target_label(labels if isinstance(labels, list) else [])
        )
    prepared = prepared.dropna(subset=["target"])
    prepared["text"] = (prepared["title"] + " " + prepared["body"]).apply(normalize_text)
    prepared = prepared[prepared["text"].str.len() > 0]

    if prepared["target"].nunique() < 2:
        raise ValueError("At least two target classes are required for training.")

    logger.info("Prepared dataset rows: %s", len(prepared))
    logger.info("Target distribution:\n%s", prepared["target"].value_counts())
    return prepared


def split_dataset(prepared: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    class_counts = prepared["target"].value_counts()
    test_size = max(0.2, len(class_counts) / len(prepared))
    test_size = min(test_size, 0.5)
    stratify = prepared["target"] if class_counts.min() >= 2 and len(prepared) * test_size >= len(class_counts) else None
    return train_test_split(
        prepared["text"],
        prepared["target"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


CV_FOLDS = 5
CV_SCORERS = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "f1": make_scorer(f1_score, average="weighted", zero_division=0),
}


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray | list[str]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def cross_validate_pipeline(pipeline: Pipeline, x: pd.Series, y: pd.Series) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, x, y, cv=cv, scoring=CV_SCORERS, n_jobs=-1)
    metrics = {
        metric: float(scores[f"test_{metric}"].mean())
        for metric in CV_SCORERS
    }
    logger.info(
        "CV(%d-fold) accuracy=%.4f f1=%.4f",
        CV_FOLDS, metrics["accuracy"], metrics["f1"],
    )
    return metrics


def train_mlp(x_train: pd.Series, x_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    min_df = 2 if len(x_train) >= 20 else 1

    def _make_pipeline() -> Pipeline:
        return Pipeline(steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=min_df)),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                max_iter=30,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )),
        ])

    metrics = cross_validate_pipeline(_make_pipeline(), x_train, y_train)
    final_model = _make_pipeline()
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, MODEL_DIR / "mlp.pkl")
    return metrics


def train_text_classifier(
    name: str,
    classifier: Any,
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    min_df = 2 if len(x_train) >= 20 else 1

    def _make_pipeline() -> Pipeline:
        return Pipeline(steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=min_df)),
            ("classifier", clone(classifier)),
        ])

    metrics = cross_validate_pipeline(_make_pipeline(), x_train, y_train)
    final_model = _make_pipeline()
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, MODEL_DIR / f"{name.lower()}.pkl")
    return metrics


def train_word_char_classifier(
    name: str,
    classifier: Any,
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    def _make_pipeline() -> Pipeline:
        return Pipeline(steps=[
            ("features", FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    analyzer="word",
                    max_features=35000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                )),
                ("char_tfidf", TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=25000,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                )),
            ])),
            ("classifier", clone(classifier)),
        ])

    metrics = cross_validate_pipeline(_make_pipeline(), x_train, y_train)
    final_model = _make_pipeline()
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, MODEL_DIR / f"{name.lower()}.pkl")
    return metrics


def train_linear_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    return {
        "LogisticRegression": train_word_char_classifier(
            "LogisticRegression",
            LogisticRegression(
                C=8.0,
                class_weight="balanced",
                max_iter=1000,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            x_train,
            x_test,
            y_train,
            y_test,
        ),
    }


def train_tree_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    return {
        "DecisionTree": train_text_classifier(
            "DecisionTree",
            DecisionTreeClassifier(max_depth=28, min_samples_leaf=3, random_state=RANDOM_STATE),
            x_train,
            x_test,
            y_train,
            y_test,
        ),
        "RandomForest": train_text_classifier(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=36,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            x_train,
            x_test,
            y_train,
            y_test,
        ),
    }


def build_cluster_data(prepared: pd.DataFrame) -> dict[str, Any]:
    sampled = (
        prepared.groupby("target", group_keys=False)
        .apply(lambda group: group.sample(n=min(CLUSTER_SAMPLE_PER_LABEL, len(group)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    cluster_text = (
        sampled["title"].fillna("").astype(str)
        + " "
        + sampled["title"].fillna("").astype(str)
        + " "
        + sampled["body"].fillna("").astype(str).str[:900]
    ).str.replace("_", " ", regex=False).apply(normalize_text)
    vectorizer = TfidfVectorizer(
        max_features=3500,
        stop_words=sorted(ENGLISH_STOP_WORDS.union(CLUSTER_STOP_WORDS)),
        min_df=3,
        max_df=0.78,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(cluster_text)

    semantic_components = min(60, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if semantic_components < 2:
        raise ValueError("At least two TF-IDF features are required for clustering.")

    svd = TruncatedSVD(n_components=semantic_components, random_state=RANDOM_STATE)
    semantic_matrix = svd.fit_transform(matrix)
    semantic_matrix = Normalizer(copy=False).fit_transform(semantic_matrix)
    coords = semantic_matrix[:, :2]

    cluster_count = min(CLUSTER_COUNT, len(sampled))
    kmeans = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=20)
    cluster_ids = kmeans.fit_predict(semantic_matrix)
    score = silhouette_score(semantic_matrix, cluster_ids) if cluster_count > 1 else 0.0

    feature_names = np.array(vectorizer.get_feature_names_out())
    points = []
    clusters = []
    for row_id, (_, row) in enumerate(sampled.reset_index(drop=True).iterrows()):
        points.append(
            {
                "id": row_id,
                "x": round(float(coords[row_id][0]), 5),
                "y": round(float(coords[row_id][1]), 5),
                "clusterId": int(cluster_ids[row_id]),
                "category": row["target"],
                "title": row["title"],
                "repo": row.get("repo", ""),
            }
        )

    for cluster_id in range(cluster_count):
        member_indices = np.where(cluster_ids == cluster_id)[0]
        if len(member_indices) == 0:
            continue
        label_distribution = sampled.iloc[member_indices]["target"].value_counts().to_dict()
        category = str(max(label_distribution, key=label_distribution.get))
        centroid = matrix[member_indices].mean(axis=0).A1
        keyword_indices = centroid.argsort()[::-1][:12]
        semantic_centroid = semantic_matrix[member_indices].mean(axis=0)
        distances = np.linalg.norm(semantic_matrix[member_indices] - semantic_centroid, axis=1)
        representative_indices = member_indices[np.argsort(distances)[:5]]
        clusters.append(
            {
                "categoryId": int(cluster_id),
                "clusterId": int(cluster_id),
                "name": f"Cluster {cluster_id + 1}",
                "dominantCategory": category,
                "count": int(len(member_indices)),
                "color": CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                "labelDistribution": {str(key): int(value) for key, value in label_distribution.items()},
                "keywords": feature_names[keyword_indices].tolist(),
                "representativeIssues": [
                    {
                        "title": str(sampled.iloc[index]["title"]),
                        "repo": str(sampled.iloc[index].get("repo", "")),
                        "url": str(sampled.iloc[index].get("url", "")),
                    }
                    for index in representative_indices
                ],
            }
        )

    return {
        "summary": {
            "totalIssues": int(len(sampled)),
            "sourceIssues": int(len(prepared)),
            "samplePerLabel": int(CLUSTER_SAMPLE_PER_LABEL),
            "clusterCount": int(cluster_count),
            "silhouetteScore": round(float(score), 4),
            "algorithm": "TF-IDF + TruncatedSVD + KMeans",
        },
        "points": points,
        "clusters": clusters,
        "colors": CLUSTER_COLORS,
    }


def require_tensorflow() -> None:
    if any(item is None for item in [Sequential, Tokenizer, pad_sequences, to_categorical]):
        raise RuntimeError("TensorFlow is required to train LSTM and GRU models.")


def vectorize_sequences(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    max_words: int = 20000,
    max_len: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    require_tensorflow()
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train.tolist())
    x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train.tolist()), maxlen=max_len)
    x_test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test.tolist()), maxlen=max_len)

    label_encoder = LabelEncoder()
    y_train_ids = label_encoder.fit_transform(y_train)
    y_test_ids = label_encoder.transform(y_test)

    y_train_cat = to_categorical(y_train_ids, num_classes=len(label_encoder.classes_))
    y_test_cat = to_categorical(y_test_ids, num_classes=len(label_encoder.classes_))
    return x_train_seq, x_test_seq, y_train_cat, y_test_cat, tokenizer, label_encoder


def build_recurrent_model(kind: str, num_classes: int, max_words: int = 20000, max_len: int = 220) -> Sequential:
    require_tensorflow()
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    if kind == "lstm":
        model.add(LSTM(96, dropout=0.2, recurrent_dropout=0.1))
    elif kind == "gru":
        model.add(GRU(96, dropout=0.2, recurrent_dropout=0.1))
    else:
        raise ValueError(f"Unsupported recurrent model: {kind}")
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_recurrent_model(
    kind: str,
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    x_train_seq, x_test_seq, y_train_cat, _y_test_cat, tokenizer, label_encoder = vectorize_sequences(
        x_train,
        x_test,
        y_train,
        y_test,
    )
    model = build_recurrent_model(kind, num_classes=len(label_encoder.classes_))
    model.fit(
        x_train_seq,
        y_train_cat,
        validation_split=0.1,
        epochs=8,
        batch_size=64,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1,
    )

    probabilities = model.predict(x_test_seq, verbose=0)
    prediction_ids = probabilities.argmax(axis=1)
    predictions = label_encoder.inverse_transform(prediction_ids)
    metrics = evaluate_predictions(y_test, predictions)

    model.save(MODEL_DIR / f"{kind}.keras")
    joblib.dump(tokenizer, MODEL_DIR / f"{kind}_tokenizer.pkl")
    joblib.dump(label_encoder, MODEL_DIR / f"{kind}_label_encoder.pkl")
    return metrics


def save_metrics(metrics: dict[str, dict[str, float]]) -> None:
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def save_cluster_data(prepared: pd.DataFrame) -> None:
    FRONTEND_LIB_DIR.mkdir(parents=True, exist_ok=True)
    cluster_data = build_cluster_data(prepared)
    with (FRONTEND_LIB_DIR / "cluster-data.json").open("w", encoding="utf-8") as file:
        json.dump(cluster_data, file, ensure_ascii=False, indent=2)


def train_all(dataset_path: Path = DATASET_PATH) -> dict[str, dict[str, float]]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_issues(dataset_path)
    prepared = prepare_dataset(df)
    x_train, x_test, y_train, y_test = split_dataset(prepared)

    metrics: dict[str, dict[str, float]] = {}
    logger.info("Training linear text models")
    metrics.update(train_linear_models(x_train, x_test, y_train, y_test))

    logger.info("Training decision tree and ensemble models")
    metrics.update(train_tree_models(x_train, x_test, y_train, y_test))

    logger.info("Training MLP model")
    metrics["MLP"] = train_mlp(x_train, x_test, y_train, y_test)

    try:
        logger.info("Training LSTM model")
        metrics["LSTM"] = train_recurrent_model("lstm", x_train, x_test, y_train, y_test)
        logger.info("Training GRU model")
        metrics["GRU"] = train_recurrent_model("gru", x_train, x_test, y_train, y_test)
    except RuntimeError as exc:
        logger.warning("%s LSTM/GRU training skipped.", exc)

    save_metrics(metrics)
    save_cluster_data(prepared)
    return metrics
