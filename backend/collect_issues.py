import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests import Response, Session
from tqdm import tqdm


GITHUB_API_URL = "https://api.github.com"
TARGET_REPOSITORIES = [
    "facebook/react",
    "vercel/next.js",
    "reduxjs/redux-toolkit",
    "pmndrs/zustand",
    "react-hook-form/react-hook-form",
    "TanStack/query",
]
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset" / "issues.json"
MAX_ISSUES_PER_REPO = 1500
PER_PAGE = 100
REQUEST_TIMEOUT = 30


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


IssueRecord = dict[str, str | int | list[str]]


LABEL_NORMALIZATION_RULES: dict[str, set[str]] = {
    "bug": {"bug", "type: bug", "kind/bug", "bug: confirmed"},
    "feature": {"enhancement", "feature request", "feature", "new feature"},
    "documentation": {"docs", "documentation", "doc"},
    "question": {"question", "support", "help wanted"},
}


@dataclass
class RepositoryStats:
    repo: str
    fetched_total: int = 0
    excluded_pr: int = 0
    excluded_no_label: int = 0
    excluded_no_normalized_label: int = 0
    excluded_no_title: int = 0
    saved: int = 0


def get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError(
            "GitHub Personal Access Token is required. "
            "Set GITHUB_TOKEN before running this script."
        )
    return token


def create_session(token: str) -> Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "react-issue-finder-ai-dataset-collector",
        }
    )
    return session


def sleep_until_rate_limit_resets(response: Response) -> None:
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")

    if remaining == "0" and reset:
        wait_seconds = max(int(reset) - int(time.time()), 0) + 5
        logger.warning("GitHub rate limit reached. Sleeping for %s seconds.", wait_seconds)
        time.sleep(wait_seconds)


def request_with_retry(session: Session, url: str, params: dict[str, Any]) -> Response:
    last_response: Response | None = None

    for attempt in range(1, 4):
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        last_response = response
        sleep_until_rate_limit_resets(response)

        if response.status_code == 403 and "rate limit" in response.text.lower():
            sleep_until_rate_limit_resets(response)
            continue

        if response.status_code in {500, 502, 503, 504}:
            wait_seconds = attempt * 5
            logger.warning(
                "GitHub API temporary error %s. Retrying in %s seconds.",
                response.status_code,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        return response

    if last_response is None:
        raise RuntimeError("GitHub request failed before receiving a response.")
    last_response.raise_for_status()
    return last_response


def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None

    for part in link_header.split(","):
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url_part = section[0].strip()
        rel_part = section[1].strip()
        if rel_part == 'rel="next"':
            return url_part.removeprefix("<").removesuffix(">")
    return None


def is_pull_request(issue: dict[str, Any]) -> bool:
    return "pull_request" in issue


def extract_labels(issue: dict[str, Any]) -> list[str]:
    return [
        label["name"].strip()
        for label in issue.get("labels", [])
        if isinstance(label, dict) and isinstance(label.get("name"), str) and label["name"].strip()
    ]


def normalize_label(label: str) -> str | None:
    normalized = label.lower().strip()
    for target, source_labels in LABEL_NORMALIZATION_RULES.items():
        if normalized in source_labels:
            return target
    return None


def normalize_labels(labels: list[str]) -> list[str]:
    normalized_labels = []
    for label in labels:
        normalized = normalize_label(label)
        if normalized and normalized not in normalized_labels:
            normalized_labels.append(normalized)
    return normalized_labels


def normalize_issue(repo_full_name: str, issue: dict[str, Any], stats: RepositoryStats) -> IssueRecord | None:
    title = issue.get("title")
    if not title:
        stats.excluded_no_title += 1
        return None

    labels = extract_labels(issue)
    if not labels:
        stats.excluded_no_label += 1
        return None

    labels_normalized = normalize_labels(labels)
    if not labels_normalized:
        stats.excluded_no_normalized_label += 1
        return None

    return {
        "repo": repo_full_name,
        "title": str(title),
        "body": issue.get("body") or "",
        "labels": labels,
        "labels_normalized": labels_normalized,
        "state": issue.get("state", ""),
        "comments": int(issue.get("comments") or 0),
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "url": issue.get("html_url", ""),
    }


def collect_repository_issues(session: Session, repo_full_name: str) -> tuple[list[IssueRecord], RepositoryStats]:
    url = f"{GITHUB_API_URL}/repos/{repo_full_name}/issues"
    params: dict[str, Any] = {
        "state": "all",
        "sort": "created",
        "direction": "desc",
        "per_page": PER_PAGE,
    }
    collected: list[IssueRecord] = []
    stats = RepositoryStats(repo=repo_full_name)
    page = 1

    logger.info("Collecting issues from %s", repo_full_name)
    progress = tqdm(total=MAX_ISSUES_PER_REPO, desc=repo_full_name, unit="issue")

    while url and len(collected) < MAX_ISSUES_PER_REPO:
        try:
            response = request_with_retry(session, url, params)
        except Exception as exc:
            logger.exception("Failed to collect %s page %s: %s", repo_full_name, page, exc)
            break

        issues = response.json()
        if not isinstance(issues, list):
            logger.warning("Unexpected response for %s page %s: %s", repo_full_name, page, issues)
            break

        stats.fetched_total += len(issues)

        for issue in issues:
            if len(collected) >= MAX_ISSUES_PER_REPO:
                break

            if is_pull_request(issue):
                stats.excluded_pr += 1
                continue

            record = normalize_issue(repo_full_name, issue, stats)
            if record:
                collected.append(record)
                stats.saved += 1
                progress.update(1)

        progress.set_postfix(
            {
                "saved": stats.saved,
                "pr": stats.excluded_pr,
                "no_label": stats.excluded_no_label,
            }
        )

        if len(collected) >= MAX_ISSUES_PER_REPO:
            logger.info("%s reached MAX_ISSUES_PER_REPO=%s", repo_full_name, MAX_ISSUES_PER_REPO)
            break

        url = parse_next_link(response.headers.get("Link"))
        params = {}
        page += 1

    progress.close()
    log_repository_stats(stats)
    return collected, stats


def log_repository_stats(stats: RepositoryStats) -> None:
    logger.info(
        (
            "%s quality stats | fetched_total=%s, excluded_pr=%s, "
            "excluded_no_label=%s, excluded_no_normalized_label=%s, "
            "excluded_no_title=%s, saved=%s"
        ),
        stats.repo,
        stats.fetched_total,
        stats.excluded_pr,
        stats.excluded_no_label,
        stats.excluded_no_normalized_label,
        stats.excluded_no_title,
        stats.saved,
    )


def save_issues(issues: list[IssueRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(issues, file, ensure_ascii=False, indent=2)


def count_normalized_labels(issues: list[IssueRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for issue in issues:
        labels = issue.get("labels_normalized", [])
        if isinstance(labels, list):
            counter.update(str(label) for label in labels)
    return counter


def print_quality_summary(issues: list[IssueRecord], stats_by_repo: list[RepositoryStats]) -> None:
    repo_counts = Counter(str(issue["repo"]) for issue in issues)
    label_counts = count_normalized_labels(issues)
    title_lengths = [len(str(issue.get("title", ""))) for issue in issues]
    body_lengths = [len(str(issue.get("body", ""))) for issue in issues]

    print("\nCollection quality summary")
    for stats in stats_by_repo:
        print(
            f"- {stats.repo}: fetched={stats.fetched_total}, "
            f"excluded_pr={stats.excluded_pr}, "
            f"excluded_no_label={stats.excluded_no_label}, "
            f"excluded_no_normalized_label={stats.excluded_no_normalized_label}, "
            f"saved={stats.saved}"
        )

    print("\nEDA summary")
    print(f"Total data count: {len(issues)}")
    print("Repository data count:")
    for repo, count in repo_counts.items():
        print(f"- {repo}: {count}")

    print("Normalized label distribution:")
    for label, count in label_counts.items():
        print(f"- {label}: {count}")

    print("Top 20 labels:")
    for label, count in label_counts.most_common(20):
        print(f"- {label}: {count}")

    avg_title_length = sum(title_lengths) / len(title_lengths) if title_lengths else 0
    avg_body_length = sum(body_lengths) / len(body_lengths) if body_lengths else 0
    print(f"Average title length: {avg_title_length:.2f}")
    print(f"Average body length: {avg_body_length:.2f}")
    print(f"Saved path: {OUTPUT_PATH}")


def main() -> None:
    token = get_github_token()
    session = create_session(token)
    all_issues: list[IssueRecord] = []
    stats_by_repo: list[RepositoryStats] = []

    for repo_full_name in TARGET_REPOSITORIES:
        repo_issues, stats = collect_repository_issues(session, repo_full_name)
        all_issues.extend(repo_issues)
        stats_by_repo.append(stats)

    save_issues(all_issues, OUTPUT_PATH)
    print_quality_summary(all_issues, stats_by_repo)


if __name__ == "__main__":
    main()
