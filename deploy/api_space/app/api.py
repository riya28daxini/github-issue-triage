"""FastAPI service for the issue triage models."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.predictor import IssueTriage, fetch_issue

state = {}


@asynccontextmanager
async def lifespan(app):
    state["triage"] = IssueTriage(artifact_dir=os.environ.get("ARTIFACT_DIR"),
                                  load_models=os.environ.get("LOAD_MODELS", "1") != "0")
    yield


app = FastAPI(title="GitHub Issue Triage", version="1.0", lifespan=lifespan,
              description="Predicts issue type, tags, and similar issues.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IssueIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    body: str = Field("", max_length=50_000)
    repo: str | None = Field(None, description="e.g. pandas-dev/pandas. Leave empty to search all supported repositories.")
    top_k: int = Field(5, ge=1, le=20)


class UrlIn(BaseModel):
    url: str
    top_k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "supported_repositories": list(state["triage"].indexes)}


@app.post("/predict")
def predict(issue: IssueIn):
    return state["triage"].predict(issue.title, issue.body, issue.repo, issue.top_k)


@app.post("/predict_url")
def predict_url(req: UrlIn):
    try:
        issue = fetch_issue(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    out = state["triage"].predict(issue["title"], issue["body"], issue["repo"], req.top_k)
    out["issue"] = {"title": issue["title"], "repo": issue["repo"], "number": issue["number"]}
    return out
