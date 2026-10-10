# Full results

Detailed scores for every model. The short summary is in the [README](../README.md).


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

**Error analysis** ([details](docs/ERROR_ANALYSIS.md)): the issue-type baseline reaches 0.96 accuracy on the test set. Its mistakes are mostly docs and feature requests worded like bugs, and many sampled errors look like ambiguous maintainer labels. `question` is missed 7 times out of 10. The models partly learn issue-template wording and title prefixes, but accuracy drops only about one point on titles without a prefix. Per-tag threshold tuning did not improve the tag results, because tag frequencies shift over time.

## Engineering decisions

- **Got past GitHub's 100-page limit.** My first collector silently stopped at 2,000 to 5,000 issues per repo (HTTP 422). I rewrote it with the Search API and date windows to reach 8,000 per repo.
- **Removed non-user traffic.** Bots and internal team items were a third of VS Code's issues and would have taught the wrong patterns.
- **Exact label matching instead of keywords**, after finding that "Needs Decision - Include Feature" is not a feature request.
- **Built a duplicate ground truth from two sources.** Comments alone revealed the original issue for only about 40% of duplicates. GitHub's GraphQL API records it when a maintainer uses "Close as duplicate", so combining both gave 699 usable pairs out of 857 duplicates.
- **Experiments are tracked with MLflow**, with a repo-name ablation to check the models are not just recognizing the repository.

