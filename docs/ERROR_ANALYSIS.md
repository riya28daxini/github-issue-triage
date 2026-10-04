# Error analysis of the TF-IDF baselines

Model for issue type: TF-IDF + LinearSVC (no repo name). Model for tags: TF-IDF + One-vs-Rest Logistic Regression. All numbers are on the time-based **test** set. Notebook: `notebooks/04_error_analysis.ipynb`.

## Issue type

2,482 test issues, **accuracy 0.96**, macro-F1 0.82.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bug | 0.97 | 0.99 | 0.98 | 1,645 |
| docs | 0.98 | 0.89 | 0.93 | 361 |
| feature | 0.94 | 0.94 | 0.94 | 466 |
| question | 0.75 | 0.30 | 0.43 | 10 |

Confusion matrix (rows = true class, columns = predicted):

| | bug | docs | feature | question |
|---|---|---|---|---|
| **bug** | 1,622 | 6 | 17 | 0 |
| **docs** | 27 | 322 | 11 | 1 |
| **feature** | 24 | 2 | 440 | 0 |
| **question** | 7 | 0 | 0 | 3 |

**Where the errors are**
- Most mistakes are issues of another type predicted as `bug` (docs to bug 27, feature to bug 24) and bugs predicted as `feature` (17).
- `question` is missed 7 times out of 10, always predicted as `bug`. In the sample I read, both missed `question` issues carried a `BUG:` title, which suggests maintainers relabeled them as usage questions after investigating (my interpretation). That judgment is not visible in the text when the issue is created, which may be why this class is hard.
- Error rate per repository: transformers 0.4%, pandas 3.7%, scikit-learn 5.8%, **VS Code 13.6%** (118 test issues, the noisiest repository).
- Per-repository macro-F1 is not reported: repositories with almost no `docs` or `question` issues get an artificially low macro-F1, so error rate is the fairer number.

**Does the model just read title prefixes?**
- 96% of pandas titles start with a prefix such as `BUG:`, `ENH:` or `DOC:`, against 1% to 10% in the other repositories.
- Accuracy is 96.7% with a prefix and 95.8% without, so prefixes help by about one point and the model works without them. The comparison is partly confounded by repository, since nearly all prefixed titles are from pandas.

**What the model learned (top words per class)**
- bug: `bug`, `codeblock`, `reproduce`, `expected`, `behavior`, `fails`
- docs: `doc`, `documentation`, `docstring`, `user guide`
- feature: `enh`, `feature request`, `motivation`, `your contribution`, `support`, `add`
- question: `how can`, `how`, `qst`, `stackoverflow`

Many of these are issue-template headings (`Motivation`, `Your contribution`, `Expected behavior`, `To reproduce`) and title prefixes (`ENH`, `QST`). That is legitimate, since the author picks the template when filing, but it means accuracy on free-form issues without a template is likely lower than 96%. A template-free evaluation is left as future work.

**Reading 15 wrong predictions**
- In my reading of this small sample, about 11 of the 15 look like ambiguous or noisy maintainer labels (a subjective judgment). Examples: "BUG: 1-element pd.NA boolean arrays are all() but not any()" is labeled `Usage Question`; "CountVectorizer fails when dealing with short strings..." is labeled `Documentation`; an "ENH/API: Architectural Audit" issue is labeled `Bug`.
- Three or four are clear model mistakes: documentation requests that are worded like problems ("Update videos list with recent presentations", "Accessibility issues in documentation website") were predicted as `bug`.

## Tags

Per-tag F1 on the test set at the default threshold, and after tuning one threshold per tag on the validation set:

| Tag | Test support | F1 default | F1 tuned |
|---|---|---|---|
| performance | 74 | 0.713 | 0.672 |
| spam_invalid | 89 | 0.643 | 0.654 |
| build_ci | 101 | 0.441 | 0.377 |
| needs_discussion | 261 | 0.354 | 0.385 |
| api | 61 | 0.333 | 0.310 |
| needs_info | 94 | 0.279 | 0.239 |
| regression | 57 | 0.245 | 0.337 |

Micro-F1 0.400 (default) vs 0.393 (tuned); macro-F1 0.430 vs 0.425.

- `performance` and `spam_invalid` are the easiest tags, because they have distinctive wording.
- `regression`, `needs_info`, `api` and `needs_discussion` are the hardest. They are often applied by maintainers after reading the discussion, not from the text of the first post.
- **Threshold tuning did not help.** It improved three tags and hurt four, with no gain overall. Tag frequencies change over time (for example `needs_info` has 151 validation but 94 test examples), so thresholds tuned on a small validation set do not transfer. The reported tag results use the default threshold.
