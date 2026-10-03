
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

DATA = Path("data/processed/issues_clean.parquet")
MODELS = Path("models")
TAGS = ["api", "build_ci", "needs_discussion", "needs_info",
        "performance", "regression", "spam_invalid"]


def make_text(df, repo_token):
    text = df["text_clean"]
    if repo_token:
        text = "REPO_" + df["repo"].str.replace(r"[^A-Za-z0-9]", "_", regex=True) + " " + text
    return text


def make_model(name):
    if name == "lr":
        return LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
    return LinearSVC(C=0.5, class_weight="balanced")


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000,
                           sublinear_tf=True, strip_accents="unicode")


def run_type(df, args):
    d = df.dropna(subset=["issue_type"])
    parts = {s: d[d["split"] == s] for s in ["train", "val", "test"]}
    classes = sorted(d["issue_type"].unique())

    vec = make_vectorizer()
    X = {"train": vec.fit_transform(make_text(parts["train"], args.repo_token))}
    for s in ["val", "test"]:
        X[s] = vec.transform(make_text(parts[s], args.repo_token))

    clf = make_model(args.model).fit(X["train"], parts["train"]["issue_type"])

    metrics, preds = {}, {}
    for s in ["val", "test"]:
        y, p = parts[s]["issue_type"], clf.predict(X[s])
        preds[s] = (y, p)
        metrics[f"{s}_macro_f1"] = f1_score(y, p, average="macro")
        metrics[f"{s}_accuracy"] = accuracy_score(y, p)
        for c, f in zip(classes, f1_score(y, p, labels=classes, average=None, zero_division=0)):
            metrics[f"{s}_f1_{c}"] = f

    # 'question' has very few val/test examples, so also report val and test combined
    if "question" in classes:
        y = pd.concat([preds["val"][0], preds["test"][0]])
        p = np.concatenate([preds["val"][1], preds["test"][1]])
        metrics["valtest_f1_question"] = f1_score(y, p, labels=["question"], average=None, zero_division=0)[0]

    y, p = preds["test"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, p, labels=classes, normalize="true",
                                            ax=ax, values_format=".2f", colorbar=False)
    ax.set_title("Test set (rows = true class)")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)

    return metrics, {
        "classification_report_test.txt": classification_report(y, p, zero_division=0),
        "n_train": len(parts["train"]),
    }, (vec, clf), "confusion_matrix.png"


def run_tags(df, args):
    d = df[df["labels"].apply(len) > 0]
    parts = {s: d[d["split"] == s] for s in ["train", "val", "test"]}
    mlb = MultiLabelBinarizer(classes=TAGS)
    Y = {s: mlb.fit_transform([list(t) for t in parts[s]["tags"]]) for s in parts}

    vec = make_vectorizer()
    X = {"train": vec.fit_transform(make_text(parts["train"], args.repo_token))}
    for s in ["val", "test"]:
        X[s] = vec.transform(make_text(parts[s], args.repo_token))

    clf = OneVsRestClassifier(make_model(args.model)).fit(X["train"], Y["train"])

    metrics = {}
    for s in ["val", "test"]:
        p = clf.predict(X[s])
        metrics[f"{s}_micro_f1"] = f1_score(Y[s], p, average="micro", zero_division=0)
        metrics[f"{s}_macro_f1"] = f1_score(Y[s], p, average="macro", zero_division=0)
        for t, f in zip(TAGS, f1_score(Y[s], p, average=None, zero_division=0)):
            metrics[f"{s}_f1_{t}"] = f

    report = classification_report(Y["test"], clf.predict(X["test"]), target_names=TAGS, zero_division=0)
    return metrics, {"classification_report_test.txt": report,
                     "n_train": len(parts["train"])}, (vec, clf, mlb), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["type", "tags"], required=True)
    ap.add_argument("--model", choices=["lr", "svm"], default="lr")
    ap.add_argument("--repo-token", action="store_true",
                    help="prepend the repo name to the text as an extra feature")
    args = ap.parse_args()

    df = pd.read_parquet(DATA)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("issue-triage")
    run_name = f"{args.task}-tfidf-{args.model}{'-repo' if args.repo_token else ''}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"task": args.task, "model": args.model, "repo_token": args.repo_token,
                           "tfidf_ngrams": "1-2", "tfidf_min_df": 3, "split": "time-based per repo"})
        runner = run_type if args.task == "type" else run_tags
        metrics, extra, model_objs, figure = runner(df, args)

        mlflow.log_param("n_train", extra["n_train"])
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_text(extra["classification_report_test.txt"], "classification_report_test.txt")
        if figure:
            mlflow.log_artifact(figure)

        MODELS.mkdir(exist_ok=True)
        joblib.dump(model_objs, MODELS / f"{run_name}.joblib")

    print(f"\n=== {run_name} ===")
    print(extra["classification_report_test.txt"])
    for k in sorted(metrics):
        if k.endswith("macro_f1") or k.endswith("micro_f1") or k.endswith("accuracy") or k.startswith("valtest"):
            print(f"{k:28s} {metrics[k]:.3f}")


if __name__ == "__main__":
    main()
