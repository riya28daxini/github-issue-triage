
import re
import sys
from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/issues_all.parquet")
OUT_PATH = Path("data/processed/issues_clean.parquet")

HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)          # issue-template instructions
CODE_BLOCK = re.compile(r"```.*?```", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@\w[\w-]*")
ISSUE_REF = re.compile(r"(?<!\w)#\d+")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
ERROR_NAME = re.compile(r"\b[A-Z]\w*(?:Error|Exception|Warning)\b")
TRACEBACK = re.compile(
    r"Traceback \(most recent call last\)|Exception in thread|^\s+at .+\(.+:\d+:\d+\)", re.M
)
MD_NOISE = re.compile(r"[#*>|_~=\-]{1,}")
SPACES = re.compile(r"\s+")

MAX_BODY_CHARS = 4000


def clean_issue(title, body):
    title = title or ""
    body = HTML_COMMENT.sub(" ", body or "")

    # facts taken from the raw body, before anything is removed
    has_code = bool(CODE_BLOCK.search(body))
    has_traceback = bool(TRACEBACK.search(body))
    n_urls = len(URL.findall(body))
    errors = sorted(set(ERROR_NAME.findall(body)))   # e.g. ValueError, KeyError: useful bug signals

    text = IMAGE.sub(" IMAGE ", body)
    text = CODE_BLOCK.sub(" CODEBLOCK ", text)
    text = MD_LINK.sub(r"\1", text)
    text = URL.sub(" URL ", text)
    text = MENTION.sub(" USER ", text)
    text = ISSUE_REF.sub(" ISSUEREF ", text)
    text = INLINE_CODE.sub(r"\1", text)
    text = MD_NOISE.sub(" ", text)
    text = SPACES.sub(" ", text).strip()[:MAX_BODY_CHARS]

    clean = f"{title.strip()} . {text} {' '.join(errors)}".strip()
    return {
        "text_clean": clean,
        "has_code_block": has_code,
        "has_traceback": has_traceback,
        "n_urls": n_urls,
        "n_error_names": len(errors),
        "title_len": len(title),
        "body_len": len(body),
    }


def main():
    if not IN_PATH.exists():
        sys.exit(f"{IN_PATH} not found. Run the last cell of 02_labels.ipynb first.")

    df = pd.read_parquet(IN_PATH)
    feats = pd.DataFrame([clean_issue(t, b) for t, b in zip(df["title"], df["body"])], index=df.index)
    df = pd.concat([df, feats], axis=1)

    empty = (df["text_clean"].str.len() < 10).sum()
    print(f"rows: {len(df)}  very short texts: {empty}")
    print(df[["has_code_block", "has_traceback"]].mean().round(2).to_dict())
    print(df["text_clean"].str.len().describe().round(0).to_dict())

    df.to_parquet(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
