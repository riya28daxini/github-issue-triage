
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPOS = [
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "microsoft/vscode",
    "huggingface/transformers",
]

SEARCH_URL = "https://api.github.com/search/issues"
SEARCH_DELAY = 2.2            # Search API allows 30 requests per minute
EARLIEST = datetime(2012, 1, 1, tzinfo=timezone.utc)
FMT = "%Y-%m-%dT%H:%M:%SZ"


def make_logger(log_file):
    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def make_session():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit('GITHUB_TOKEN not set. In PowerShell run: $env:GITHUB_TOKEN = Read-Host "Paste token"')
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return s


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


def search(session, repo, start, end, page, log, max_retries=6):
    """Return (json, error). error is None on success."""
    query = f"repo:{repo} is:issue created:{start.strftime(FMT)}..{end.strftime(FMT)}"
    params = {"q": query, "per_page": 100, "page": page, "sort": "created", "order": "desc"}

    for attempt in range(1, max_retries + 1):
        time.sleep(SEARCH_DELAY)
        try:
            resp = session.get(SEARCH_URL, params=params, timeout=40)
        except requests.RequestException as e:
            log(f"network error ({e}), retry {attempt}/{max_retries}")
            time.sleep(10 * attempt)
            continue

        if resp.status_code == 200:
            data = resp.json()
            if data.get("incomplete_results"):
                log(f"warning: incomplete results for {start:%Y-%m-%d}..{end:%Y-%m-%d} page {page}")
            return data, None

        if resp.status_code in (403, 429):
            if resp.headers.get("Retry-After"):
                wait = int(resp.headers["Retry-After"]) + 5
            elif resp.headers.get("X-RateLimit-Remaining") == "0" and resp.headers.get("X-RateLimit-Reset"):
                wait = max(int(resp.headers["X-RateLimit-Reset"]) - time.time(), 0) + 5
            else:
                wait = 60 * attempt
            log(f"HTTP {resp.status_code}, sleeping {wait:.0f}s, retry {attempt}/{max_retries}")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            log(f"server error {resp.status_code}, retry {attempt}/{max_retries}")
            time.sleep(10 * attempt)
            continue

        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

    return None, f"gave up after {max_retries} retries"


def fetch_window(session, repo, start, end, f, budget, seen, log):
    """Write issues created in [start, end]. Splits the window if it has more than 1000 results."""
    data, err = search(session, repo, start, end, 1, log)
    if err:
        raise RuntimeError(err)
    total = data["total_count"]
    if total == 0:
        return 0

    if total > 1000 and (end - start) > timedelta(hours=1):
        half = int((end - start).total_seconds() // 2)
        mid = start + timedelta(seconds=half)
        written = fetch_window(session, repo, mid + timedelta(seconds=1), end, f, budget, seen, log)  # newer half first
        if written < budget:
            written += fetch_window(session, repo, start, mid, f, budget - written, seen, log)
        return written

    written, page = 0, 1
    while True:
        for issue in data["items"]:
            if "pull_request" in issue or issue["number"] in seen:
                continue
            seen.add(issue["number"])
            f.write(json.dumps(simplify(issue, repo), ensure_ascii=False) + "\n")
            written += 1
            if written >= budget:
                return written
        if len(data["items"]) < 100 or page >= 10:
            break
        page += 1
        data, err = search(session, repo, start, end, page, log)
        if err:
            raise RuntimeError(err)
    return written


def collect_repo(session, repo, max_issues, out_dir, log):
    out_path = out_dir / (repo.replace("/", "__") + ".jsonl")
    count, seen = 0, set()
    reason = "reached --max-per-repo"
    end = datetime.now(timezone.utc)
    log(f"Starting {repo} (target {max_issues})")

    with open(out_path, "w", encoding="utf-8") as f:
        try:
            while count < max_issues and end > EARLIEST:
                start = max(end - timedelta(days=90), EARLIEST)
                n = fetch_window(session, repo, start, end, f, max_issues - count, seen, log)
                count += n
                log(f"{repo}: {start:%Y-%m-%d} to {end:%Y-%m-%d}: +{n}, total {count}")
                end = start - timedelta(seconds=1)
        except RuntimeError as e:
            reason = f"ERROR: {e}"

    if count < max_issues and not reason.startswith("ERROR"):
        reason = f"went back to {EARLIEST:%Y} (start of search range)"
    log(f"FINISHED {repo}: {count} issues saved to {out_path}. Stopped because: {reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-repo", type=int, default=8000)
    parser.add_argument("--out-dir", default="data/raw_full")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "collection_log.txt")
    session = make_session()

    for repo in REPOS:
        collect_repo(session, repo, args.max_per_repo, out_dir, log)
    log("ALL DONE")


if __name__ == "__main__":
    main()