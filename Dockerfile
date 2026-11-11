FROM python:3.11-slim

LABEL maintainer="AI Engineering Team"
LABEL description="Model Regression Detection System (LLM CI/CD Quality Pipeline)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source, prompts, data, and configs
COPY src/ ./src/
COPY data/ ./data/
COPY prompts/ ./prompts/
COPY thresholds.yaml .

# Create volume/directory for reports and persistent SQLite runs
RUN mkdir -p /app/report /app/data

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--candidate", "v2", "--baseline", "v1", "--out", "report"]
