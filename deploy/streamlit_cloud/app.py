"""Streamlit demo for the GitHub issue triage project. Loads the models from the Hugging Face Hub and runs them directly."""
import os

import pandas as pd
import requests
import streamlit as st

from predictor import IssueTriage, fetch_issue


def get_secret(name, default):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return default


REPOS = ["(search all)", "pandas-dev/pandas", "scikit-learn/scikit-learn", "microsoft/vscode", "huggingface/transformers"]
EXAMPLES = {
    "(write your own)": ("", "", "(search all)"),
    "Bug report (pandas)": (
        "groupby with observed=False raises KeyError on categorical column",
        "### Reproducible Example\n```python\nimport pandas as pd\ndf = pd.DataFrame({'a': pd.Categorical(['x', 'y']), 'b': [1, 2]})\n"
        "df.groupby('a', observed=False)['c'].sum()\n```\n### Issue Description\nKeyError: 'c' is raised, expected a clearer message.\n"
        "### Expected Behavior\nA helpful error message.", "pandas-dev/pandas"),
    "Feature request (scikit-learn)": (
        "Add sample_weight support to the OneHotEncoder",
        "### Describe the workflow you want to enable\nI would like to pass sample weights when computing category frequencies, "
        "for example to drop rare categories.\n### Describe your proposed solution\nAdd a `sample_weight` argument to `fit`.", "scikit-learn/scikit-learn"),
    "Documentation (pandas)": (
        "DOC: typo in the docstring of DataFrame.merge",
        "The docstring of `DataFrame.merge` says 'suffixs' instead of 'suffixes' in the parameter description.", "pandas-dev/pandas"),
}

st.set_page_config(page_title="GitHub Issue Triage", page_icon="🧭", layout="wide")
st.title("🧭 GitHub Issue Triage")
st.caption("Predicts the issue type and tags, finds similar existing issues, and estimates the attention an issue may get. "
           "Trained on 29,000+ issues from scikit-learn, pandas, VS Code and Hugging Face Transformers.")


@st.cache_resource(show_spinner="Loading the models (the first load takes a minute or two)...")
def get_triage():
    return IssueTriage(artifact_dir=os.environ.get("ARTIFACT_DIR"),
                       repo_id=get_secret("ARTIFACT_REPO", "Riyaaa28/issue-triage-artifacts"),
                       load_models=os.environ.get("LOAD_MODELS", "1") != "0")


def run_text(title, body, repo):
    return get_triage().predict(title, body, repo, top_k=5)


def run_url(url):
    try:
        issue = fetch_issue(url)
    except (ValueError, RuntimeError) as e:
        st.error(str(e))
        return None
    res = get_triage().predict(issue["title"], issue["body"], issue["repo"], 5,
                               exclude=(issue["repo"], issue["number"]))
    res["issue"] = {"title": issue["title"], "repo": issue["repo"], "number": issue["number"]}
    return res


def show(res):
    left, right = st.columns(2)
    with left:
        st.subheader("Issue type")
        st.success(res["issue_type"]["label"].upper())
        st.bar_chart(pd.Series(res["issue_type"]["scores"]))
        st.caption(res["issue_type"]["model"])

        st.subheader("Tags")
        predicted = res["tags"]["predicted"]
        st.write(" ".join(f"`{t}`" for t in predicted) if predicted else "No tag above 50%.")
        st.bar_chart(pd.Series(res["tags"]["probabilities"]))
        st.caption(res["tags"]["model"])
    with right:
        st.subheader("Similar existing issues")
        st.caption(f"Searched: {res['searched_repo']}")
        sims = pd.DataFrame(res["similar_issues"])
        if sims.empty:
            st.write("No similar issues found.")
        else:
            st.dataframe(sims[["repo", "number", "title", "similarity", "url"]], hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("Open", display_text="open")})
        st.subheader("Estimated attention level (experimental)")
        att = res["attention_level"]
        st.info(att["label"].upper())
        st.bar_chart(pd.Series(att["probabilities"]))
        st.caption(att["note"])


tab_text, tab_url = st.tabs(["Paste an issue", "From a GitHub link"])

with tab_text:
    example = st.selectbox("Start from an example", list(EXAMPLES))
    ex_title, ex_body, ex_repo = EXAMPLES[example]
    title = st.text_input("Title", value=ex_title, key=f"title-{example}")
    body = st.text_area("Body", value=ex_body, height=220, key=f"body-{example}")
    repo = st.selectbox("Repository to search for similar issues", REPOS, index=REPOS.index(ex_repo), key=f"repo-{example}")
    if st.button("Analyze issue", type="primary"):
        if not title.strip():
            st.warning("Please enter a title.")
        else:
            with st.spinner("Analyzing..."):
                res = run_text(title, body, None if repo == REPOS[0] else repo)
            if res:
                show(res)

with tab_url:
    url = st.text_input("Link to a public issue", placeholder="https://github.com/pandas-dev/pandas/issues/12345")
    if st.button("Fetch and analyze"):
        with st.spinner("Fetching the issue and analyzing..."):
            res = run_url(url)
        if res:
            st.write(f"**{res['issue']['repo']} #{res['issue']['number']}**: {res['issue']['title']}")
            show(res)

st.divider()
st.caption("Attention level is an experimental proxy for comments and reactions, not an official priority. "
           "Similar-issue search covers the four repositories in the training data. "
           "[Source code and write-up on GitHub](https://github.com/riya28daxini/github-issue-triage)")
