import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "issues.json"
TARGET_LABELS = {"bug", "feature", "documentation", "question"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


REQUIRED_FIELDS = {
    "repo",
    "title",
    "body",
    "labels",
    "labels_normalized",
    "state",
    "comments",
    "created_at",
    "updated_at",
    "url",
}


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset root must be a JSON array.")

    return data


def validate_dataset(data: list[dict[str, Any]]) -> pd.DataFrame:
    errors: list[str] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Row {index}: item is not an object")
            continue

        missing = REQUIRED_FIELDS - set(item)
        if missing:
            errors.append(f"Row {index}: missing fields {sorted(missing)}")

        if not item.get("title"):
            errors.append(f"Row {index}: empty title")

        if not isinstance(item.get("body"), str):
            errors.append(f"Row {index}: body must be a string")

        labels = item.get("labels")
        if not isinstance(labels, list) or not labels:
            errors.append(f"Row {index}: labels must be a non-empty list")
        elif not all(isinstance(label, str) for label in labels):
            errors.append(f"Row {index}: labels must contain only strings")

        normalized = item.get("labels_normalized")
        if not isinstance(normalized, list) or not normalized:
            errors.append(f"Row {index}: labels_normalized must be a non-empty list")
        elif not all(isinstance(label, str) and label in TARGET_LABELS for label in normalized):
            errors.append(f"Row {index}: labels_normalized contains invalid labels")

        if not isinstance(item.get("comments"), int):
            errors.append(f"Row {index}: comments must be an integer")

    if errors:
        logger.warning("JSON validation found %s issue(s).", len(errors))
        for error in errors[:50]:
            logger.warning(error)
        if len(errors) > 50:
            logger.warning("Additional validation errors omitted: %s", len(errors) - 50)
    else:
        logger.info("JSON validation passed. Rows: %s", len(data))

    df = pd.DataFrame(data)
    for column in REQUIRED_FIELDS:
        if column not in df.columns:
            df[column] = [[] for _ in range(len(df))] if column in {"labels", "labels_normalized"} else ""
    return df


def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_length"] = df["title"].fillna("").astype(str).str.len()
    df["body_length"] = df["body"].fillna("").astype(str).str.len()
    return df


def count_list_column(df: pd.DataFrame, column: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for values in df[column]:
        if isinstance(values, list):
            counter.update(str(value) for value in values)
    return counter


def normalized_label_quality(df: pd.DataFrame) -> dict[str, int]:
    empty_count = 0
    invalid_count = 0
    multi_label_count = 0

    for labels in df["labels_normalized"]:
        if not isinstance(labels, list) or not labels:
            empty_count += 1
            continue
        if any(label not in TARGET_LABELS for label in labels):
            invalid_count += 1
        if len(labels) > 1:
            multi_label_count += 1

    return {
        "empty_labels_normalized": empty_count,
        "invalid_labels_normalized": invalid_count,
        "multi_label_rows": multi_label_count,
    }


def evaluate_training_readiness(df: pd.DataFrame, labels: Counter[str]) -> None:
    total_rows = len(df)
    available_classes = [label for label in TARGET_LABELS if labels[label] > 0]
    min_class_count = min((labels[label] for label in available_classes), default=0)
    can_train = total_rows >= 100 and len(available_classes) >= 2 and min_class_count >= 10

    print("\nAI training readiness")
    print(f"- Total rows: {total_rows}")
    print(f"- Available classes: {available_classes}")
    print(f"- Minimum class count: {min_class_count}")
    print(f"- Trainable: {'YES' if can_train else 'NO'}")

    if not can_train:
        print("- Recommendation: collect more labeled issues, especially for classes with low counts.")


def print_tables(df: pd.DataFrame, raw_labels: Counter[str], normalized_labels: Counter[str]) -> None:
    print("\nRepository issue counts")
    print(df["repo"].value_counts())

    print("\nNormalized label distribution")
    normalized_df = pd.DataFrame(normalized_labels.most_common(), columns=["label", "count"])
    print(normalized_df)

    print("\nTop 20 raw labels")
    raw_df = pd.DataFrame(raw_labels.most_common(20), columns=["label", "count"])
    print(raw_df)

    print("\nLength and comment summary")
    print(df[["title_length", "body_length", "comments"]].describe())

    print("\nlabels_normalized quality")
    for key, value in normalized_label_quality(df).items():
        print(f"- {key}: {value}")


def plot_bar(counter: Counter[str], title: str, xlabel: str, ylabel: str, top_n: int | None = None) -> None:
    items = counter.most_common(top_n)
    if not items:
        logger.warning("No data to plot for %s", title)
        return

    plot_df = pd.DataFrame(items, columns=[xlabel, ylabel])
    plot_df.plot(kind="bar", x=xlabel, y=ylabel, legend=False, title=title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_repository_counts(df: pd.DataFrame) -> None:
    repo_counts = Counter(df["repo"])
    plot_bar(repo_counts, "Repository별 Issue 개수", "Repository", "Issue Count")


def plot_label_distribution(labels: Counter[str]) -> None:
    plot_bar(labels, "정규화 Label 분포", "Label", "Count")


def plot_top_raw_labels(labels: Counter[str]) -> None:
    plot_bar(labels, "원본 Label Top 20", "Label", "Count", top_n=20)


def plot_histogram(df: pd.DataFrame, column: str, title: str, bins: int = 50) -> None:
    df[column].plot(kind="hist", bins=bins, title=title)
    plt.xlabel(column)
    plt.ylabel("Issue Count")
    plt.tight_layout()
    plt.show()


def run_eda(path: Path = DATASET_PATH) -> None:
    data = load_dataset(path)
    df = validate_dataset(data)
    if df.empty:
        raise ValueError("Dataset is empty.")

    df = add_length_columns(df)
    raw_labels = count_list_column(df, "labels")
    normalized_labels = count_list_column(df, "labels_normalized")

    print_tables(df, raw_labels, normalized_labels)
    evaluate_training_readiness(df, normalized_labels)
    plot_label_distribution(normalized_labels)
    plot_top_raw_labels(raw_labels)
    plot_repository_counts(df)
    plot_histogram(df, "title_length", "제목 길이 분포")
    plot_histogram(df, "body_length", "본문 길이 분포")
    plot_histogram(df, "comments", "댓글 수 분포")


if __name__ == "__main__":
    run_eda()
