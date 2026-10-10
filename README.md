# GitHub Issue Triage

An NLP system that triages GitHub issues: it predicts the **issue type**, suggests **tags**, finds **likely duplicates** and estimates **priority**. Trained on 29,000+ real issues I collected from scikit-learn, pandas, VS Code and Hugging Face Transformers.

**Live demo: [open the app](https://app-issue-triage-hptj62dn5bryucgri55ufy.streamlit.app/)** (free hosting: the first load after a pause takes a minute or two)

![Demo screenshot](docs/demo.png)

> **Status:** complete: data pipeline, models, live demo and a GitHub Action bot.

## What it does

| Output | Task | Approach |
|---|---|---|
| Issue type: bug / feature / docs / question | Multi-class classification | TF-IDF baselines vs fine-tuned DistilBERT |
| Tags: performance, build_ci, api, regression, needs_info, needs_discussion, spam_invalid | Multi-label classification | One-vs-Rest baselines vs transformer head |
| Similar existing issues | Semantic retrieval | Sentence-BERT + FAISS |
| Priority: low / medium / high | Classification | XGBoost on features known at creation time |

## Data at a glance

- **29,201 real user issues** (14,878 with a usable type label) after removing bots, Copilot and team-internal items
- **Time-based split** per repository (oldest 70% train, next 15% validation, newest 15% test), so the model is always tested on future issues
- Priority is a **proxy** built from comments and reactions, used only as a label and never as an input (no leakage)
- Full details: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

## Results so far

**Issue type**

| Model | Val macro-F1 | Test macro-F1 | Question F1 (val + test) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.848 | 0.797 | 0.581 |
| TF-IDF + LinearSVC | 0.863 | 0.820 | 0.628 |
| TF-IDF + Logistic Regression + repo name | 0.850 | 0.804 | 0.587 |
| TF-IDF + LinearSVC + repo name | 0.865 | 0.822 | 0.635 |
| DistilBERT (fine-tuned, 5 epochs) | 0.811 | 0.794 | 0.421 |

**Tags (multi-label)**

| Model | Val micro-F1 | Test micro-F1 | Test macro-F1 |
|---|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression | 0.359 | 0.400 | 0.430 |
| TF-IDF + One-vs-Rest LinearSVC | 0.372 | 0.400 | 0.425 |
| DistilBERT multi-label (5 epochs) | 0.404 | 0.439 | 0.471 |

LinearSVC beats Logistic Regression by about 0.02 macro-F1, and adding the repository name changes scores by less than 0.01. The `question` class is rare (10 test examples), so it is scored on validation and test combined. Tag prediction is much harder than issue type.

**DistilBERT vs the baselines:** for issue type, DistilBERT matches the TF-IDF model on `bug`, `docs` and `feature` (test F1 0.98 / 0.94 / 0.95 vs 0.98 / 0.93 / 0.94) but is weaker on the rare `question` class, so the TF-IDF LinearSVC is used for issue type. For tags DistilBERT is better (test micro-F1 0.439 vs 0.400, better on 5 of 7 tags), so it is used for tags.

**Duplicate retrieval.** For 636 issues closed as duplicates (699 pairs found, minus 63 whose original is newer than the duplicate), the system searches earlier issues of the same repository, as a real bot would (median pool: 5,424 candidates).

| Method | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| TF-IDF cosine | 0.138 | 0.215 | 0.263 | 0.182 |
| Sentence-BERT (all-MiniLM-L6-v2) | 0.239 | 0.436 | 0.531 | 0.338 |

Sentence-BERT roughly doubles the baseline. Showing 10 suggestions finds the original about half the time. The pool is smaller than a repository's full history, so a real deployment would score lower.

**Priority** (low / medium / high, an engagement proxy). XGBoost on the Sentence-BERT embedding plus creation-time features reaches test macro-F1 0.392, against 0.378 for metadata alone and 0.240 for always predicting "low". Accuracy (0.468) is below the trivial "always low" baseline (0.563) because the model is trained to balance the classes. Priority is therefore treated as an experimental score: predicting community engagement from the first post is hard. Used only as a ranking score it is weakly informative: AUC 0.62 for spotting the top 15% of issues by engagement, and a Spearman correlation of 0.21 with actual engagement.

**Error analysis** ([details](docs/ERROR_ANALYSIS.md)): the issue-type baseline reaches 0.96 accuracy on the test set. Its mistakes are mostly docs and feature requests worded like bugs, and many sampled errors look like ambiguous maintainer labels. `question` is missed 7 times out of 10. The models partly learn issue-template wording and title prefixes, but accuracy drops only about one point on titles without a prefix. Per-tag threshold tuning did not improve the tag results, because tag frequencies shift over time.

## Engineering decisions worth noting

- **Got past GitHub's 100-page limit.** My first collector silently stopped at 2,000 to 5,000 issues per repo (HTTP 422). I rewrote it with the Search API and date windows to reach 8,000 per repo.
- **Removed non-user traffic.** Bots and internal team items were a third of VS Code's issues and would have taught the wrong patterns.
- **Exact label matching instead of keywords**, after finding that "Needs Decision - Include Feature" is not a feature request.
- **Built a duplicate ground truth from two sources.** Comments alone revealed the original issue for only about 40% of duplicates. GitHub's GraphQL API records it when a maintainer uses "Close as duplicate", so combining both gave 699 usable pairs out of 857 duplicates.
- **Experiments are tracked with MLflow**, with a repo-name ablation to check the models are not just recognizing the repository.

## Try it

- **Live:** the link at the top. Paste an issue (or a link to a public issue from scikit-learn, pandas, VS Code or Transformers) to see its type, tags, similar existing issues and an experimental attention level.
- **Deployment note:** Hugging Face no longer offers free Docker Spaces, so the live demo runs on Streamlit Community Cloud (`deploy/streamlit_cloud`). The FastAPI + Docker version (`deploy/api_space`) is included and was tested locally.
- **Run it yourself:** `pip install -r deploy/streamlit_cloud/requirements.txt`, then `streamlit run deploy/streamlit_cloud/app.py`. The models download automatically from the Hugging Face Hub (`Riyaaa28/issue-triage-artifacts`).

## The bot

A GitHub Action (`.github/workflows/triage.yml`) runs the same models whenever an issue is opened in this repository. It comments with the predicted type and tags, similar existing issues and an experimental attention level, and adds labels. It runs inside the Actions runner, so no server is needed.

![Bot comment](docs/bot_comment.png)

It skips issues opened by bots, reads the issue from GitHub's event file instead of the shell, and neutralizes `@mentions` in the suggested titles.

## Quick start

```bash
git clone https://github.com/riya28daxini/github-issue-triage.git
cd github-issue-triage
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/collect_issues.py --max-per-repo 8000 --out-dir data/raw   # needs a GITHUB_TOKEN environment variable
# run notebooks/02_labels.ipynb (Restart & Run All)
python src/preprocess.py
python src/train_baseline.py --task type --model svm
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Data files are not stored in the repository; the collector recreates them.

## Tech stack

Python, pandas, scikit-learn, MLflow, PyTorch and Hugging Face Transformers, Sentence-Transformers, FAISS, XGBoost, FastAPI, Streamlit, Docker.

## Author

**Riya**, B.Tech Computer Engineering, Nirma University | [LinkedIn](https://www.linkedin.com/in/riya-daxini-623627377) | [GitHub](https://github.com/riya28daxini)
