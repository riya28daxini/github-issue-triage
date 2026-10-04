"""
Build the duplicate ground truth: for every issue that was closed as a duplicate, find the ORIGINAL issue.

Sources, in this order:
  1. GitHub GraphQL: ClosedEvent.duplicateOf / MarkedAsDuplicateEvent.canonical
     (this is what the "Close as duplicate" button records; works for VS Code, transformers, scikit-learn)
  2. A comment such as "duplicate of #123"  (needed for pandas, which uses comments)
Then it downloads the original issues that are not in our dataset yet, so retrieval can find them.

Setup: secure token setup from the header of probe_duplicates.py (the token is never typed into a command).

Run from the project root (about 20 to 30 minutes; safe to stop and restart, progress is saved):
    python src/build_duplicate_pairs.py

Outputs (in data/processed/):
    duplicate_raw.jsonl     one line per duplicate issue checked (progress file)
    duplicate_pairs.csv     repo, number (the duplicate), target (the original), method
    extra_corpus.jsonl      original issues that were not in issues_all.parquet
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from collect_issues import simplify  # same record format as the main dataset

REST = "https://api.github.com"
GQL = "https://api.github.com/graphql"
DATA = Path("data/processed")
RAW = DATA / "duplicate_raw.jsonl"
PAIRS = DATA / "duplicate_pairs.csv"
EXTRA = DATA / "extra_corpus.jsonl"

REF = "__typename ... on Issue { number repository { nameWithOwner } } ... on PullRequest { number repository { nameWithOwner } }"
QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      timelineItems(first: 30, itemTypes: [CLOSED_EVENT, MARKED_AS_DUPLICATE_EVENT]) {
        nodes {
          __typename
          ... on ClosedEvent { stateReason duplicateOf { %s } }
          ... on MarkedAsDuplicateEvent { canonical { %s } }
        }
      }
    }
  }
}
""" % (REF, REF)

ISSUE_URL = r"https?://github\.com/[\w.-]+/[\w.-]+/issues/"
PATTERNS = [
    re.compile(rf"(?i)duplicate\s+of\s+(?:{ISSUE_URL}|[\w.-]+/[\w.-]+#|#)(\d+)"),
    re.compile(rf"(?i)(?:dup(?:licate)?s?|same as|covered (?:by|in))\D{{0,40}}?(?:{ISSUE_URL}|#)(\d+)"),
]


def extract_from_text(text, own_number):
    for pat in PATTERNS:
        for m in pat.findall(text or ""):
            if int(m) != own_number:
                return int(m)
    return None


def make_session():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN not set (see the setup note at the top of probe_duplicates.py).")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}",
                      "Accept": "application/vnd.github+json",
                      "X-GitHub-Api-Version": "2022-11-28"})
    return s


def request(session, method, url, retries=5, **kw):
    """HTTP call with retries and rate-limit waiting. Returns the response or None."""
    for attempt in range(1, retries + 1):
        try:
            r = session.request(method, url, timeout=40, **kw)
        except requests.RequestException:
            time.sleep(5 * attempt)
            continue
        if r.status_code in (200, 404, 410):
            return r
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset")
            wait = max(int(reset) - time.time(), 0) + 5 if r.headers.get("X-RateLimit-Remaining") == "0" and reset else 60 * attempt
            print(f"  rate limited, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        time.sleep(5 * attempt)       # 5xx and anything else
    return None


def graphql_target(session, repo, number):
    owner, name = repo.split("/")
    r = request(session, "POST", GQL, json={"query": QUERY,
                                            "variables": {"owner": owner, "name": name, "number": number}})
    if r is None or r.status_code != 200:
        return None
    data = r.json()
    issue = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}
    for node in (issue.get("timelineItems") or {}).get("nodes") or []:
        ref = node.get("duplicateOf") if node.get("__typename") == "ClosedEvent" else node.get("canonical")
        if ref and ref.get("number") and (ref.get("repository") or {}).get("nameWithOwner") == repo:
            return int(ref["number"])
    return None


def comment_target(session, repo, number):
    r = request(session, "GET", f"{REST}/repos/{repo}/issues/{number}/comments", params={"per_page": 100})
    if r is None or r.status_code != 200:
        return None
    for c in r.json():
        t = extract_from_text(c.get("body"), number)
        if t is not None:
            return t
    return None


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    session = make_session()
    df = pd.read_parquet(DATA / "issues_all.parquet")
    have = set(zip(df["repo"], df["number"].astype(int)))

    dup = df[(df["state_reason"] == "duplicate") | df["is_duplicate_labeled"]][["repo", "number"]].drop_duplicates()
    print(f"{len(dup)} duplicate issues to check:\n{dup.groupby('repo').size().to_string()}\n", flush=True)

    done = set()
    if RAW.exists():
        with open(RAW, encoding="utf-8") as f:
            done = {(j["repo"], j["number"]) for j in map(json.loads, f)}
        print(f"resuming: {len(done)} already checked\n", flush=True)

    with open(RAW, "a", encoding="utf-8") as out:
        for i, (repo, number) in enumerate(zip(dup["repo"], dup["number"].astype(int)), 1):
            if (repo, number) in done:
                continue
            target, method = graphql_target(session, repo, number), "graphql"
            if target is None:
                target, method = comment_target(session, repo, number), "comment"
            out.write(json.dumps({"repo": repo, "number": number,
                                  "target": target, "method": method if target else None}) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"  checked {i}/{len(dup)}", flush=True)
            time.sleep(0.15)

    raw = pd.DataFrame([json.loads(l) for l in open(RAW, encoding="utf-8")]).drop_duplicates(["repo", "number"], keep="last")
    pairs = raw.dropna(subset=["target"]).copy()
    pairs["target"] = pairs["target"].astype(int)

    # download originals that are not in the dataset yet
    extra_rows, bad = [], set()
    if EXTRA.exists():
        extra_rows = [json.loads(l) for l in open(EXTRA, encoding="utf-8")]
        have |= {(r["repo"], r["number"]) for r in extra_rows}
    missing = sorted({(r, t) for r, t in zip(pairs["repo"], pairs["target"]) if (r, t) not in have})
    print(f"\n{len(pairs)} pairs found; downloading {len(missing)} original issues not in the dataset", flush=True)
    with open(EXTRA, "a", encoding="utf-8") as out:
        for repo, t in missing:
            r = request(session, "GET", f"{REST}/repos/{repo}/issues/{t}")
            if r is None or r.status_code != 200 or "pull_request" in r.json():
                bad.add((repo, t))                    # missing, deleted, or a pull request
                continue
            rec = simplify(r.json(), repo)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            have.add((repo, t))
            time.sleep(0.15)

    pairs = pairs[[(r, t) in have for r, t in zip(pairs["repo"], pairs["target"])]]
    pairs.to_csv(PAIRS, index=False)

    print("\n=== duplicate pairs usable for evaluation ===")
    print(pairs.groupby(["repo", "method"]).size().unstack(fill_value=0).to_string())
    print(f"\ntotal pairs: {len(pairs)}   (originals unavailable: {len(bad)})")
    print(f"saved {PAIRS} and {EXTRA}")


if __name__ == "__main__":
    main()
