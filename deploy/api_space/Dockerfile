FROM python:3.11-slim

WORKDIR /code
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV ARTIFACT_REPO=Riyaaa28/issue-triage-artifacts
ENV HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
