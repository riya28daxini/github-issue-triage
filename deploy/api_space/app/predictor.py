"""
Issue triage predictor.

Loads every model from one Hugging Face Hub repository and, for a new issue, returns:
  - issue type      (TF-IDF + LinearSVC)
  - tags            (fine-tuned DistilBERT, multi-label)
  - similar issues  (Sentence-BERT + FAISS, one index per repository)
  - attention level (XGBoost; an experimental proxy for community engagement)
"""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from app.preprocess import clean_issue

ATTENTION_NOTE = ("Experimental: estimates how much community attention an issue may get "
                  "(comments and reactions), not an official maintainer priority.")
ISSUE_URL = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/issues/(\d+)")


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def fetch_issue(url):
    """Download a public GitHub issue (title, body, repository) from its URL."""
    m = ISSUE_URL.search(url or "")
    if not m:
        raise ValueError("Please paste a link like https://github.com/owner/repo/issues/123")
    owner, name, number = m.groups()
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    r = requests.get(f"https://api.github.com/repos/{owner}/{name}/issues/{number}", headers=headers, timeout=20)
    if r.status_code == 404:
        raise ValueError("Issue not found (is the repository public?)")
    if r.status_code in (403, 429):
        raise RuntimeError("GitHub rate limit reached. Paste the title and body instead.")
    r.raise_for_status()
    j = r.json()
    if "pull_request" in j:
        raise ValueError("That link is a pull request, not an issue.")
    return {"title": j["title"], "body": j.get("body") or "", "repo": f"{owner}/{name}", "number": int(number)}


class IssueTriage:
    def __init__(self, artifact_dir=None, repo_id=None, load_models=True):
        if artifact_dir is None:
            from huggingface_hub import snapshot_download
            artifact_dir = snapshot_download(repo_id or os.environ["ARTIFACT_REPO"])
        d = Path(artifact_dir)

        labels = json.load(open(d / "labels.json"))
        self.types, self.tags = labels["types"], labels["tags"]
        self.max_len_tags = labels.get("max_len_tags", 256)
        self.embedding_model = labels["embedding_model"]

        self.type_vec, self.type_clf = joblib.load(d / "type_tfidf_svm.joblib")

        import xgboost as xgb
        self.prio_cfg = json.load(open(d / "priority_config.json"))
        self.prio = xgb.XGBClassifier()
        self.prio.load_model(str(d / "priority_xgb.json"))

        import faiss
        self.meta = pd.read_parquet(d / "corpus_meta_slim.parquet")
        self.indexes = {}
        for repo in self.prio_cfg["repo_cats"]:
            name = repo.replace("/", "__")
            self.indexes[repo] = (faiss.read_index(str(d / f"faiss_{name}.index")), np.load(d / f"faiss_{name}_rows.npy"))

        if load_models:
            self._load_heavy(d)

    def _load_heavy(self, d):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.tag_tok = AutoTokenizer.from_pretrained(str(d / "tags_distilbert"))
        self.tag_model = AutoModelForSequenceClassification.from_pretrained(str(d / "tags_distilbert")).eval()
        self.embedder = SentenceTransformer(self.embedding_model, device="cpu")
        self.embedder.max_seq_length = 256

    # the two methods that need the neural models (kept separate so tests can replace them)
    def _embed(self, text):
        return self.embedder.encode([text], normalize_embeddings=True)[0].astype("float32")

    def _tag_probs(self, text):
        enc = self.tag_tok(text, truncation=True, max_length=self.max_len_tags, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.tag_model(**enc).logits[0].numpy()
        return 1 / (1 + np.exp(-logits))

    def _similar(self, emb, repo, top_k):
        repos = [repo] if repo in self.indexes else list(self.indexes)
        hits = []
        for r in repos:
            index, rows = self.indexes[r]
            scores, ids = index.search(emb[None, :], top_k)
            for s, i in zip(scores[0], ids[0]):
                if i < 0:
                    continue
                row = self.meta.iloc[int(rows[i])]
                hits.append({"repo": row["repo"], "number": int(row["number"]), "title": str(row["title"]),
                             "similarity": round(float(s), 3),
                             "url": f"https://github.com/{row['repo']}/issues/{int(row['number'])}"})
        return sorted(hits, key=lambda h: -h["similarity"])[:top_k]

    def _attention(self, feats, emb, repo):
        cfg = self.prio_cfg
        assoc = cfg["assoc_cats"].index("NONE") if "NONE" in cfg["assoc_cats"] else -1
        repo_code = cfg["repo_cats"].index(repo) if repo in cfg["repo_cats"] else -1
        x = [float(feats[c]) for c in cfg["meta_cols"]] + [float(assoc), float(repo_code)]
        x = np.array(x, dtype=float)
        if cfg["uses_embeddings"]:
            x = np.concatenate([x, emb.astype(float)])
        proba = self.prio.predict_proba(x[None, :])[0]
        levels = cfg["levels"]
        return {"label": levels[int(np.argmax(proba))],
                "probabilities": {l: round(float(p), 3) for l, p in zip(levels, proba)},
                "note": ATTENTION_NOTE}

    def predict(self, title, body="", repo=None, top_k=5):
        c = clean_issue(title, body)
        text = c["text_clean"]

        scores = self.type_clf.decision_function(self.type_vec.transform([text]))[0]
        type_probs = _softmax(scores)
        classes = list(self.type_clf.classes_)

        tag_probs = self._tag_probs(text)
        emb = self._embed(text)

        return {
            "issue_type": {"label": classes[int(np.argmax(scores))],
                           "scores": {k: round(float(p), 3) for k, p in zip(classes, type_probs)},
                           "model": "TF-IDF + LinearSVC (scores are relative, not calibrated probabilities)"},
            "tags": {"predicted": [t for t, p in zip(self.tags, tag_probs) if p > 0.5],
                     "probabilities": {t: round(float(p), 3) for t, p in zip(self.tags, tag_probs)},
                     "model": "DistilBERT multi-label"},
            "similar_issues": self._similar(emb, repo, top_k),
            "attention_level": self._attention(c, emb, repo),
            "searched_repo": repo if repo in self.indexes else "all supported repositories",
        }
