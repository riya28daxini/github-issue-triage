

# issue type: exactly one type must match, otherwise the issue is dropped as ambiguous
TYPE_MAP = {
    "bug": "bug",
    "regression": "bug",
    "recent-regression": "bug",
    "feature request": "feature",
    "feature-request": "feature",
    "new feature": "feature",
    "enhancement": "feature",
    "new model": "feature",
    "documentation": "docs",
    "docs": "docs",
    "usage": "question",
    "usage question": "question",
    "question": "question",
    "*question": "question",
}

# unified multi-label tags that mean the same thing across repos
TAG_MAP = {
    "performance": "performance",
    "build": "build_ci",
    "build / ci": "build_ci",
    "api": "api",
    "api design": "api",
    "regression": "regression",
    "recent-regression": "regression",
    "needs info": "needs_info",
    "info-needed": "needs_info",
    "needs reproducible code": "needs_info",
    "needs discussion": "needs_discussion",
    "needs decision": "needs_discussion",
    "needs decision - include feature": "needs_discussion",
    "rfc": "needs_discussion",
    "spam": "spam_invalid",
    "invalid": "spam_invalid",
}

# ground truth for duplicate-detection evaluation (Week 3)
DUPLICATE_LABELS = {"*duplicate", "duplicate report", "duplicate"}

# vscode team-internal tracking items (test plans, release notes), not real user reports
INTERNAL_LABELS = {"testplan-item", "on-testplan", "new release"}


def _norm(labels):
    return {l.strip().lower() for l in labels}


def map_type(labels):
    types = {TYPE_MAP[l] for l in _norm(labels) if l in TYPE_MAP}
    return next(iter(types)) if len(types) == 1 else None


def map_tags(labels):
    return sorted({TAG_MAP[l] for l in _norm(labels) if l in TAG_MAP})


def is_duplicate(labels):
    return bool(_norm(labels) & DUPLICATE_LABELS)


def is_internal(labels):
    return bool(_norm(labels) & INTERNAL_LABELS)


def apply_labels(df):
    """Add issue_type, tags, is_duplicate_labeled and is_internal columns to the DataFrame."""
    df = df.copy()
    df["issue_type"] = df["labels"].apply(map_type)
    df["tags"] = df["labels"].apply(map_tags)
    df["is_duplicate_labeled"] = df["labels"].apply(is_duplicate)
    df["is_internal"] = df["labels"].apply(is_internal)
    return df
