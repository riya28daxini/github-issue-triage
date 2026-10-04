# GitHub Issue Triage

An NLP system that triages GitHub issues: it predicts the **issue type**, suggests **tags**, finds **likely duplicates** and estimates **priority**. Trained on 29,000+ real issues I collected from scikit-learn, pandas, VS Code and Hugging Face Transformers.

> **Status: in progress** (data, labels and TF-IDF baselines done; transformers, duplicate search, API and deployment next). **Live demo:** _coming soon_

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
| DistilBERT (fine-tuned) | _planned_ | _planned_ | _planned_ |

**Tags (multi-label)**

| Model | Val micro-F1 | Test micro-F1 | Test macro-F1 |
|---|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression | 0.359 | 0.400 | 0.430 |
| TF-IDF + One-vs-Rest LinearSVC | 0.372 | 0.400 | 0.425 |
| DistilBERT multi-label | _planned_ | _planned_ | _planned_ |

LinearSVC beats Logistic Regression by about 0.02 macro-F1, and adding the repository name changes scores by less than 0.01. The `question` class is rare (10 test examples), so it is scored on validation and test combined. Tag prediction is much harder than issue type.

**Error analysis** ([details](docs/ERROR_ANALYSIS.md)): the issue-type baseline reaches 0.96 accuracy on the test set. Its mistakes are mostly docs and feature requests worded like bugs, and many sampled errors look like ambiguous maintainer labels. `question` is missed 7 times out of 10. The models partly learn issue-template wording and title prefixes, but accuracy drops only about one point on titles without a prefix. Per-tag threshold tuning did not improve the tag results, because tag frequencies shift over time.

## Engineering decisions worth noting

- **Got past GitHub's 100-page limit.** My first collector silently stopped at 2,000 to 5,000 issues per repo (HTTP 422). I rewrote it with the Search API and date windows to reach 8,000 per repo.
- **Removed non-user traffic.** Bots and internal team items were a third of VS Code's issues and would have taught the wrong patterns.
- **Exact label matching instead of keywords**, after finding that "Needs Decision - Include Feature" is not a feature request.
- **Built a duplicate ground truth from two sources.** Comments alone revealed the original issue for only about 40% of duplicates. GitHub's GraphQL API records it when a maintainer uses "Close as duplicate", so combining both gave 699 usable pairs out of 857 duplicates.
- **Experiments are tracked with MLflow**, with a repo-name ablation to check the models are not just recognizing the repository.

## Project status

- [x] Data collection (32,000 issues from 4 repositories)
- [x] Exploratory data analysis
- [x] Label mapping, filtering, priority label, time-based split
- [x] Text cleaning and TF-IDF baselines (experiments tracked with MLflow)
- [ ] Baseline error analysis
- [ ] Fine-tuned DistilBERT for type and tags
- [ ] Duplicate detection (Sentence-BERT + FAISS) and evaluation
- [ ] Priority model
- [ ] FastAPI backend and Streamlit demo
- [ ] Docker and deployment (Hugging Face Spaces)
- [ ] GitHub Action bot that comments on new issues
- [ ] Error analysis and final write-up

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

**Riya**, B.Tech Computer Engineering, Nirma University | [LinkedIn](#) | [GitHub](https://github.com/riya28daxini)
