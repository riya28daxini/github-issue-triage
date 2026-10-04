"""
Probe 2: does GitHub's GraphQL API tell us which issue a "closed as duplicate" issue points to?

The REST API did not show it (most missed issues had no comment about duplicates). GraphQL has
extra fields on timeline events that might. I have NOT run this against GitHub, so the first run
is also a test: if a field name is wrong, the script prints GitHub's error message instead of crashing.

Setup: same secure token setup as probe_duplicates.py (see the header of that file).

Run from the project root, after probe_duplicates.py has created data/processed/duplicate_probe.csv:
    python src/probe_duplicates_graphql.py

Output: printed summary + data/processed/duplicate_probe_graphql.csv
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.github.com/graphql"
IN_PATH = Path("data/processed/duplicate_probe.csv")
OUT_PATH = Path("data/processed/duplicate_probe_graphql.csv")

TARGET = "__typename ... on Issue { number repository { nameWithOwner } } ... on PullRequest { number repository { nameWithOwner } }"
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
""" % (TARGET, TARGET)


def parse_targets(data):
    """Return a list of (kind, number, repo) found in a GraphQL response."""
    found = []
    issue = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}
    for node in (issue.get("timelineItems") or {}).get("nodes") or []:
        ref = None
        if node.get("__typename") == "ClosedEvent":
            ref, kind = node.get("duplicateOf"), "closed.duplicateOf"
        elif node.get("__typename") == "MarkedAsDuplicateEvent":
            ref, kind = node.get("canonical"), "marked.canonical"
        if ref and ref.get("number"):
            found.append((kind, ref["number"], (ref.get("repository") or {}).get("nameWithOwner")))
    return found


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN not set (see probe_duplicates.py for the secure setup).")
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found. Run src/probe_duplicates.py first.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    probe = pd.read_csv(IN_PATH)

    rows, shown_raw = [], 0
    for _, r in probe.iterrows():
        owner, name = r["repo"].split("/")
        resp = session.post(URL, json={"query": QUERY,
                                       "variables": {"owner": owner, "name": name, "number": int(r["number"])}},
                            timeout=30)
        data = resp.json() if resp.status_code == 200 else {"errors": [{"message": f"HTTP {resp.status_code}: {resp.text[:200]}"}]}

        if data.get("errors"):
            print("GraphQL error:", data["errors"][0].get("message"))
            print("(If it names a field, that field does not exist; send me this message.)")
            return

        targets = parse_targets(data)
        if shown_raw < 2:                      # show the raw structure of the first responses
            print("raw example:", json.dumps(data, indent=1)[:700])
            shown_raw += 1
        gql = targets[0][1] if targets else None
        rows.append({"repo": r["repo"], "number": int(r["number"]), "comment_target": r["target"],
                     "graphql_target": gql, "graphql_kind": targets[0][0] if targets else None,
                     "graphql_repo": targets[0][2] if targets else None})
        time.sleep(0.3)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    both = out.dropna(subset=["comment_target", "graphql_target"])
    print("\n=== GraphQL found a target ===")
    print(out.groupby("repo")["graphql_target"].agg(sampled="size", found=lambda s: int(s.notna().sum())).to_string())
    print("\n=== agreement with the comment-based target (where both exist) ===")
    print(f"{int((both['comment_target'] == both['graphql_target']).sum())} of {len(both)} agree")
    print("\n=== disagreements (check these by hand) ===")
    print(both[both["comment_target"] != both["graphql_target"]].to_string())
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
