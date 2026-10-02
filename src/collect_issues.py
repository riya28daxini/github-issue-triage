"""
Collect GitHub issues (not pull requests) from selected repos into JSONL files.

Setup:
    pip install requests tqdm
    export GITHUB_TOKEN=your_personal_access_token   (Windows: set GITHUB_TOKEN=...)

Run:
    python src/collect_issues.py --max-per-repo 8000

Output: data/raw/<owner>__<repo>.jsonl  (one issue per line)
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

REPOS = [
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "microsoft/vscode",
    "huggingface/transformers",
]

API = "https://api.github.com"
OUT_DIR = Path("data/raw")


def get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN first (GitHub > Settings > Developer settings > Tokens).")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def wait_if_rate_limited(resp):
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
    if remaining <= 1:
        reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        sleep_for = max(reset - time.time(), 0) + 5
        print(f"Rate limit reached. Sleeping {sleep_for:.0f}s...")
        time.sleep(sleep_for)


def simplify(issue, repo):
    reactions = issue.get("reactions") or {}
    return {
        "repo": repo,
        "number": issue["number"],
        "title": issue["title"],
        "body": issue.get("body") or "",
        "labels": [l["name"] for l in issue.get("labels", [])],
        "state": issue["state"],
        "state_reason": issue.get("state_reason"),
        "created_at": issue["created_at"],
        "closed_at": issue.get("closed_at"),
        "comments": issue.get("comments", 0),
        "reactions_total": reactions.get("total_count", 0),
        "author": (issue.get("user") or {}).get("login"),
        "author_association": issue.get("author_association"),
        "url": issue["html_url"],
    }


def collect_repo(repo, max_issues, headers):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (repo.replace("/", "__") + ".jsonl")
    count, page = 0, 1

    with open(out_path, "w", encoding="utf-8") as f, tqdm(total=max_issues, desc=repo) as bar:
        while count < max_issues:
            resp = requests.get(
                f"{API}/repos/{repo}/issues",
                headers=headers,
                params={"state": "all", "per_page": 100, "page": page,
                        "sort": "created", "direction": "desc"},
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text[:200]}")
                break

            items = resp.json()
            if not items:
                break

            for issue in items:
                if "pull_request" in issue:  # the issues endpoint also returns PRs; skip them
                    continue
                f.write(json.dumps(simplify(issue, repo), ensure_ascii=False) + "\n")
                count += 1
                bar.update(1)
                if count >= max_issues:
                    break

            wait_if_rate_limited(resp)
            page += 1

    print(f"Saved {count} issues to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-repo", type=int, default=5000)
    args = parser.parse_args()

    headers = get_headers()
    for repo in REPOS:
        collect_repo(repo, args.max_per_repo, headers)


if __name__ == "__main__":
    main()
