# Methodology and dataset details

Details behind the numbers in the [README](../README.md): how the data was collected and labeled, how text is cleaned, how the baselines are trained and evaluated, and what the known limitations are.

## Dataset

**Source:** GitHub REST/Search API, collected on 3 October 2026. Pull requests are excluded.

| Repository | Issues collected | Date range covered |
|---|---|---|
| scikit-learn/scikit-learn | 8,000 | Aug 2017 to Oct 2026 |
| pandas-dev/pandas | 8,000 | May 2021 to Oct 2026 |
| huggingface/transformers | 8,000 | Apr 2023 to Oct 2026 |
| microsoft/vscode | 8,000 | Apr 2026 to Oct 2026 |
| **Total** | **32,000** | |

Each record has the title, body, labels, state, close reason, timestamps, comment and reaction counts, author, and author association.

**After cleaning: 29,201 real user issues**, of which **14,878** have a usable issue-type label.

### Filtering

Raw data from real projects contains a lot of noise. I removed:
- **Bot and Copilot issues** (for example `vs-code-engineering[bot]`, `Copilot`).
- **Team-created issues in VS Code** (authors with association MEMBER, COLLABORATOR or OWNER). The VS Code team uses the issue tracker for internal planning, which makes up a large share of its recent issues.
- **Internal tracking items** (VS Code labels such as `testplan-item`, `on-testplan`, `new release`).

This removed 2,799 issues (2,700 of them from VS Code).

### Labels

Maintainer labels differ in every repository, so I mapped them to a shared scheme using **exact label names** (`src/labels.py`). I first tried keyword matching and dropped it: for example, scikit-learn's `Needs Decision - Include Feature` contains the word "feature" but is not a feature request.

**Issue type** (an issue is kept only if exactly one type matches):

| Type | Source labels (examples) | Count |
|---|---|---|
| bug | `bug`, `Bug`, `Regression` | 8,752 |
| feature | `Feature request`, `New Feature`, `Enhancement`, `New model`, `feature-request` | 3,747 |
| docs | `Documentation`, `Docs` | 1,905 |
| question | `Usage`, `Usage Question`, `question` | 474 |

**Tags** (multi-label, unified across repositories):

| Tag | Merged from (examples) | Train | Val | Test |
|---|---|---|---|---|
| needs_info | `Needs Info`, `info-needed`, `Needs Reproducible Code` | 759 | 151 | 94 |
| needs_discussion | `Needs Discussion`, `Needs Decision`, `RFC` | 606 | 215 | 261 |
| regression | `Regression`, `recent-regression` | 453 | 34 | 57 |
| build_ci | `Build`, `Build / CI` | 325 | 100 | 101 |
| performance | `Performance` | 325 | 65 | 74 |
| api | `API`, `API Design` | 235 | 61 | 61 |
| spam_invalid | `spam`, `invalid` | 223 | 64 | 89 |

Process labels such as `Needs Triage`, `stale`, `WIP` and `Closing Candidate` are deliberately excluded, because predicting them would not help a maintainer.

**Priority** is a proxy, not an official maintainer label. For each issue I compute an engagement score from comments and reactions, rank it within its own repository, and bucket it: bottom 50% = low, next 35% = medium, top 15% = high. Only issues older than 14 days are scored, so recent issues are not unfairly rated low. Comments and reactions are used **only to build the label**, never as model inputs, because they accumulate after an issue is posted and would leak the answer.

| low | medium | high | not scored (too recent) |
|---|---|---|---|
| 14,702 | 9,486 | 4,327 | 686 |

**Duplicates:** 857 issues were closed as a duplicate or carry a duplicate label (638 VS Code, 191 pandas, 15 transformers, 13 scikit-learn). For each one I looked for the original issue, first in GitHub's own "closed as duplicate" record (GraphQL `ClosedEvent.duplicateOf`), then in comments such as "duplicate of #123". The original was found for **699 issues (82%)**: VS Code 535, pandas 138, scikit-learn 13, transformers 13. Pandas maintainers mostly state duplicates in comments, while VS Code, transformers and scikit-learn mostly use GitHub's button. Originals that were not already in the dataset (165) were downloaded so that retrieval can find them. These pairs evaluate duplicate retrieval, with two caveats: they are 77% VS Code, and the search corpus is smaller than each repository's full history, so recall@k is optimistic compared with a real deployment.

### Train / validation / test split

The split is **time-based within each repository** (oldest 70% train, next 15% validation, newest 15% test), not random. This matches real use, where a model is applied to future issues, and avoids leaking later information into training.

| | Train | Val | Test |
|---|---|---|---|
| Issue-type set (14,878) | 9,499 | 2,897 | 2,482 |
| Tag set (20,935) | 13,761 | 3,781 | 3,393 |
| Priority set (28,515) | 20,439 | 4,381 | 3,695 |

Issue-type classes per split:

| | bug | feature | docs | question |
|---|---|---|---|---|
| Train | 5,257 | 2,585 | 1,232 | 425 |
| Val | 1,850 | 696 | 312 | 39 |
| Test | 1,645 | 466 | 361 | 10 |

## Text preprocessing

`src/preprocess.py` turns raw issue text into model input. GitHub issues are messy (templates, logs, links), so the cleaning is specific to them:

- removes HTML comments (issue templates are full of instructions that every issue shares)
- replaces code blocks, URLs, @mentions, images and `#123` references with tokens (`CODEBLOCK`, `URL`, `USER`, `IMAGE`, `ISSUEREF`)
- strips markdown symbols and caps the body at 4,000 characters
- keeps error names found anywhere in the issue (for example `ValueError`, `KeyError`) as plain words, because they are strong bug signals

It also computes features that are **available at the moment an issue is created**: `has_code_block`, `has_traceback`, `n_urls`, `n_error_names`, `title_len`, `body_len`. These are reserved for the priority model.

## Baseline models

Before using transformers I built classical baselines (`src/train_baseline.py`), so there is a fair reference point.

| Setting | Value |
|---|---|
| Features | TF-IDF on cleaned text, word 1 to 2-grams, `min_df=3`, up to 300,000 features, sublinear term frequency |
| Issue type | Logistic Regression (C=4) and LinearSVC (C=0.5), both with balanced class weights |
| Tags | One-vs-Rest versions of the same two models, default decision threshold |
| Ablation | Optionally prepend the repository name as an extra feature, to test whether the model relies on knowing which repo an issue came from |

**Evaluation.** Accuracy is misleading here (bugs are about 59% of typed issues), so the main metric is **macro-F1**, with per-class F1 and a normalized confusion matrix on the test set. Because the `question` class has only 10 test examples, it is also scored on validation and test combined. For tags I report micro-F1, macro-F1 and per-tag F1.

## Experiment tracking

Every run is logged with MLflow (parameters, metrics, classification report, confusion matrix), so models can be compared without relying on memory.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Known limitations

- **VS Code covers only about 3 months** of recent activity, while scikit-learn goes back to 2017. VS Code also has no `docs` examples and only 14 `question` examples.
- **The `question` class is rare and drifting.** Maintainers now use that label much less, so the test set contains only 10 such issues. For this class I report scores on validation and test combined, and treat the number as indicative only.
- **`docs` and `question` come almost entirely from pandas and scikit-learn.** The model may learn repo-specific habits. I test this with an ablation that includes or excludes the repo name as a feature.
- **Labels are noisy.** They come from maintainers, are applied inconsistently, and about 40% of VS Code issues have no labels at all.
- **Priority is a proxy** based on community engagement, not on a maintainer's judgment.

## Lessons learned while building the dataset

1. **GitHub's list endpoint stops after about 100 pages.** My first collector silently ended at 2,000 to 5,000 issues per repository with an HTTP 422 error. I rewrote it to use the Search API with date windows (splitting any window with more than 1,000 results), which reached 8,000 per repository.
2. **Real data contains non-user traffic.** Bots and internal team items made up a third of VS Code's issues, and would have taught the model the wrong patterns.
3. **Exact label matching beats keyword matching** when label names are free text.
4. **Leakage can hide in plain sight.** Engagement counts look like natural priority features, but they only exist after the issue has been discussed.

## Project structure

```
github-issue-triage/
├── src/
│   ├── collect_issues.py     # GitHub Search API collector
│   ├── build_duplicate_pairs.py  # finds the original issue for each duplicate
│   ├── labels.py             # label mapping, filters, duplicate labels
│   ├── preprocess.py         # text cleaning and creation-time features
│   └── train_baseline.py     # TF-IDF baselines with MLflow tracking
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory analysis
│   ├── 02_labels.ipynb       # labels, filtering, priority, split
│   └── 03_baselines.ipynb    # baseline results and error analysis
├── docs/                     # figures
├── requirements.txt
└── README.md
```

