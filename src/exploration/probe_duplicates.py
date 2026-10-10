
import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

API = "https://api.github.com"
IN_PATH = Path("data/processed/issues_all.parquet")
OUT_PATH = Path("data/processed/duplicate_probe.csv")

ISSUE_URL = r"https?://github\.com/[\w.-]+/[\w.-]+/issues/"
PATTERNS = [
    re.compile(rf"(?i)duplicate\s+of\s+(?:{ISSUE_URL}|[\w.-]+/[\w.-]+#|#)(\d+)"),
    re.compile(rf"(?i)(?:dup(?:licate)?s?|same as|covered (?:by|in))\D{{0,40}}?(?:{ISSUE_URL}|#)(\d+)"),
]


def extract_target(text, own_number):
    """Return the first referenced issue number in a 'duplicate' sentence, or None."""
    for pat in PATTERNS:
        for m in pat.findall(text or ""):
            n = int(m)
            if n != own_number:
                return n
    return None


def get(session, url, params=None, retries=4):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(5 * attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            time.sleep(30 * attempt)
            continue
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo", type=int, default=15)
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN not set (see the setup lines at the top of this file).")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}",
                            "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28"})

    df = pd.read_parquet(IN_PATH)
    dup = df[(df["state_reason"] == "duplicate") | df["is_duplicate_labeled"]]
    print(dup.groupby("repo").size().to_string(), "\n")

    have = set(zip(df["repo"], df["number"].astype(int)))   # issues we already have in the dataset
    rows, event_types, marked_example = [], Counter(), None
    for repo, g in dup.groupby("repo"):
        sample = g.sample(min(args.per_repo, len(g)), random_state=0)
        for _, r in sample.iterrows():
            n = int(r["number"])
            comments = get(session, f"{API}/repos/{repo}/issues/{n}/comments", {"per_page": 100}) or []
            timeline = get(session, f"{API}/repos/{repo}/issues/{n}/timeline", {"per_page": 100}) or []
            for ev in timeline:
                event_types[ev.get("event", "?")] += 1
                if ev.get("event") == "marked_as_duplicate" and marked_example is None:
                    marked_example = ev

            target, snippet = None, ""
            dup_snips = [(c.get("body") or "")[:250].replace("\n", " ")
                         for c in comments if re.search(r"(?i)dup", c.get("body") or "")]
            for c in comments:
                body = c.get("body") or ""
                t = extract_target(body, n)
                if t is not None:
                    target, snippet = t, body[:200].replace("\n", " ")
                    break
            # fallback: the issue's own body sometimes says "Duplicate of #N"
            if target is None:
                t = extract_target(r.get("body") or "", n)
                if t is not None:
                    target, snippet = t, "(found in issue body)"
            rows.append({"repo": repo, "number": n, "n_comments": len(comments),
                         "target": target, "snippet": snippet, "url": r.get("url"),
                         "target_in_dataset": (repo, target) in have if target is not None else None,
                         "dup_comments": " || ".join(dup_snips[:3])})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("=== extraction rate per repo ===")
    print(out.groupby("repo")["target"].agg(sampled="size", found=lambda s: s.notna().sum()).to_string())
    print("\n=== timeline event types seen ===")
    print(dict(event_types))
    print("\n=== examples that were found ===")
    print(out.dropna(subset=["target"]).groupby("repo").head(3)[["repo", "number", "target", "snippet"]].to_string())
    print("\n=== how many found targets are already in our dataset? ===")
    print(out.dropna(subset=["target"]).groupby("repo")["target_in_dataset"].agg(found="size", in_dataset=lambda x: int(x.astype(bool).sum())).to_string())

    print("\n=== example of a 'marked_as_duplicate' timeline event (to see what fields it has) ===")
    print(json.dumps(marked_example, indent=1)[:900] if marked_example else "none seen")

    print("\n=== issues where NO target was found: what do their 'dup' comments say? ===")
    for _, m in out[out["target"].isna()].iterrows():
        print(f"[{m['repo']}] #{m['number']}  comments={m['n_comments']}")
        print("    ", m["dup_comments"][:500] if m["dup_comments"] else "(no comment mentions 'dup')")

    print(f"\nsaved {OUT_PATH}  (open it and check a few rows by hand)")


if __name__ == "__main__":
    main()