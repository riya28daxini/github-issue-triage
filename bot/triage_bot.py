"""
GitHub Action bot: comments on a newly opened issue with the predicted type, tags and similar issues.

It runs inside the GitHub Actions runner (free for public repositories), so no server is needed.
The models are downloaded from the Hugging Face Hub (Riyaaa28/issue-triage-artifacts).

Local dry run (prints the comment, posts nothing):
    $env:GITHUB_EVENT_PATH = "bot/sample_event.json"
    $env:DRY_RUN = "1"
    python bot/triage_bot.py
"""
import json
import os
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "streamlit_cloud"))
from predictor import IssueTriage  # noqa: E402

ARTIFACT_REPO = os.environ.get("ARTIFACT_REPO", "Riyaaa28/issue-triage-artifacts")
API = "https://api.github.com"
ZWSP = "\u200b"
# Links go through redirect.github.com (the same trick Dependabot uses): a plain github.com issue link would add a
# "mentioned this issue" entry to the timeline of every suggested issue in other projects.
NO_BACKLINK = "https://redirect.github.com/"


def load_issue():
    path = os.environ.get("GITHUB_EVENT_PATH")
    event = json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else {}
    issue = event.get("issue")
    if issue:                                                   # a real "issue opened" event
        return {"repo": event["repository"]["full_name"], "number": issue["number"], "title": issue["title"],
                "body": issue.get("body") or "", "author_type": (issue.get("user") or {}).get("type")}
    inputs = event.get("inputs") or {}                          # manual test run: nothing is posted
    return {"repo": os.environ.get("GITHUB_REPOSITORY"), "number": None,
            "title": inputs.get("title") or "", "body": inputs.get("body") or "", "author_type": "User"}


def safe(text, limit=110):
    """Titles come from other repositories: stop @mentions and #123 references from pinging or linking."""
    text = " ".join(str(text).split())[:limit]
    text = text.replace("@", "@" + ZWSP)
    return re.sub(r"#(\d)", "#" + ZWSP + r"\1", text)


def build_comment(res):
    tags = res["tags"]["predicted"]
    att = res["attention_level"]
    lines = ["### 🧭 Automated issue triage (experimental)", "",
             f"**Issue type:** `{res['issue_type']['label']}`",
             "**Tags:** " + (" ".join(f"`{t}`" for t in tags) if tags else "none above 50% confidence"),
             f"**Estimated attention level:** `{att['label']}` _(an experimental proxy for comments and reactions)_"]
    if res["similar_issues"]:
        lines += ["", "**Similar existing issues** (from scikit-learn, pandas, VS Code and Transformers, "
                      "the repositories this model was trained on):", ""]
        for h in res["similar_issues"]:
            link = h["url"].replace("https://github.com/", NO_BACKLINK, 1)
            lines.append(f"- [{h['repo']} #{ZWSP}{h['number']}]({link}): {safe(h['title'])} (similarity {h['similarity']:.2f})")
    lines += ["", "<sub>Suggestions only, from [github-issue-triage](https://github.com/riya28daxini/github-issue-triage). "
                  "A maintainer makes the decision.</sub>"]
    return "\n".join(lines)


def post(repo, number, body, labels):
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}
    r = requests.post(f"{API}/repos/{repo}/issues/{number}/comments", headers=headers, json={"body": body}, timeout=30)
    r.raise_for_status()
    if labels:
        r2 = requests.post(f"{API}/repos/{repo}/issues/{number}/labels", headers=headers, json={"labels": labels}, timeout=30)
        if r2.status_code >= 400:
            print("could not add labels:", r2.status_code, r2.text[:200])


def main():
    issue = load_issue()
    if issue["author_type"] == "Bot":
        print("Issue was opened by a bot; skipping.")
        return
    if not issue["title"].strip():
        sys.exit("No issue title found (for a manual run, fill in the title input).")

    triage = IssueTriage(artifact_dir=os.environ.get("ARTIFACT_DIR"), repo_id=ARTIFACT_REPO,
                         load_models=os.environ.get("LOAD_MODELS", "1") != "0")
    res = triage.predict(issue["title"], issue["body"], repo=None, top_k=5)     # search all four known repositories
    comment = build_comment(res)
    print(comment)

    if issue["number"] is None or os.environ.get("DRY_RUN") == "1":
        print("\n[dry run: nothing was posted]")
        return
    labels = []
    if os.environ.get("ADD_LABELS", "1") == "1":
        labels = [f"type: {res['issue_type']['label']}"] + [f"tag: {t}" for t in res["tags"]["predicted"]]
    post(issue["repo"], issue["number"], comment, labels)
    print("comment posted")


if __name__ == "__main__":
    main()
