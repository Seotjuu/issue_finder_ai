import json
import logging
import logging.config
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from training.predictor import IssueCategoryPredictor
except ModuleNotFoundError:
    from backend.training.predictor import IssueCategoryPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
ISSUES_PATH = DATA_DIR / "dataset" / "issues.json"
METRICS_PATH = DATA_DIR / "saved_models" / "metrics.json"
CLUSTER_DATA_PATH = DATA_DIR.parent / "frontend" / "lib" / "cluster-data.json"

ModelName = Literal[
    "LogisticRegression",
    "ComplementNB",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "MLP",
    "LSTM",
    "GRU",
]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)


class SimilarIssue(BaseModel):
    title: str
    repo: str
    similarity: float
    url: str


class ModelMetric(BaseModel):
    model: ModelName
    accuracy: float
    precision: float
    recall: float
    f1: float


class AnalyzeResponse(BaseModel):
    category: Literal["bug", "feature", "question", "documentation"]
    confidence: float
    similar_issues: list[SimilarIssue]
    model_metrics: list[ModelMetric]


class ClusterPoint(BaseModel):
    name: str
    x: float
    y: float
    count: int
    color: str


class ClusterResponse(BaseModel):
    silhouette_score: float
    clusters: list[ClusterPoint]


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    category: Literal["bug", "feature", "question", "documentation"]
    confidence: float


class SimilarityEngine:
    def __init__(self, issues: list[dict]) -> None:
        self.issues = issues
        texts = [
            f"{issue['title']} {(issue.get('body') or '')[:500]}"
            for issue in issues
        ]
        self.vectorizer = TfidfVectorizer(max_features=15000, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF ready: %d issues × %d features", self.matrix.shape[0], self.matrix.shape[1])

    def find_similar(self, text: str, top_k: int = 3) -> list[SimilarIssue]:
        vec = self.vectorizer.transform([text])
        scores = cosine_similarity(vec, self.matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SimilarIssue(
                title=self.issues[i]["title"],
                repo=self.issues[i]["repo"],
                similarity=round(float(scores[i]), 4),
                url=self.issues[i]["url"],
            )
            for i in top_indices
        ]


def _load_issues() -> list[dict]:
    with ISSUES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _load_metrics() -> list[ModelMetric]:
    with METRICS_PATH.open(encoding="utf-8") as f:
        raw: dict = json.load(f)
    return [
        ModelMetric(
            model=name,
            accuracy=vals["accuracy"],
            precision=vals["precision"],
            recall=vals["recall"],
            f1=vals["f1"],
        )
        for name, vals in raw.items()
        if name
        in (
            "LogisticRegression",
            "ComplementNB",
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "MLP",
            "LSTM",
            "GRU",
        )
    ]


app = FastAPI(title="Frontend Issue Finder AI API")
predictor = IssueCategoryPredictor()
_issues = _load_issues()
similarity_engine = SimilarityEngine(_issues)
model_metrics = _load_metrics()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    category, confidence = predictor.predict(payload.text)
    similar = similarity_engine.find_similar(payload.text)

    print(
        f"\nInput:\n  {payload.text}\n"
        f"Predicted:\n  {category} ({confidence:.2f})\n"
        f"Top Similar Issues:\n"
        + "\n".join(f"  {s.similarity:.4f}  {s.title}" for s in similar)
    )

    return AnalyzeResponse(
        category=category,
        confidence=round(confidence, 4),
        similar_issues=similar,
        model_metrics=model_metrics,
    )


@app.get("/api/clusters", response_model=ClusterResponse)
def clusters() -> ClusterResponse:
    with CLUSTER_DATA_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    return ClusterResponse(
        silhouette_score=raw["summary"]["silhouetteScore"],
        clusters=[
            ClusterPoint(
                name=cluster.get("name", f"Cluster {cluster['clusterId'] + 1}"),
                x=0,
                y=0,
                count=cluster["count"],
                color=cluster.get("color", ""),
            )
            for cluster in raw["clusters"]
        ],
    )


@app.get("/api/cluster-data")
def cluster_data() -> dict:
    with CLUSTER_DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    category, confidence = predictor.predict(payload.text)
    return PredictResponse(category=category, confidence=round(confidence, 4))
