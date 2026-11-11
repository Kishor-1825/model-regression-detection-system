# Model Regression Detection System (CI/CD for LLMs)

> **Continuous Integration and Automated Regression Prevention for LLM Features**  
> A production-grade evaluation pipeline that continuously benchmarks LLM-powered features against a versioned, hand-labeled golden dataset whenever a prompt, model, or configuration changes. It computes statistical accuracy drops, detects long-term quality drift, posts rich diff reports and PR comments, alerts on Slack, and **blocks pull request merges on critical regressions**.

---

## 1. Quick Start & Setup

### Prerequisites
- Python 3.11+
- Git
- (Optional) OpenAI API Key (system includes a deterministic offline mock mode for testing without API keys)

### Installation
```bash
# 1. Clone repository and navigate to folder
cd "Model Regression Detection System"

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment variables
copy .env.example .env
# Edit .env with your OPENAI_API_KEY and SLACK_WEBHOOK_URL
```

---

## 2. Running Evaluations

### Standard Evaluation Run
Compares candidate prompt (`v2`) against baseline (`v1`):
```bash
python -m src.cli --candidate v2 --baseline v1
```

### Force Deterministic Mock Mode (No OpenAI credits needed)
```bash
python -m src.cli --candidate v2 --baseline v1 --mock
```

### Intentionally Trigger a Regression & Test Merge Blocking
To observe how the pipeline detects a severe degradation and exits with code `1`:
```bash
python -m src.cli --candidate v3_regression --baseline v1 --mock
```
*Output: Notice the `🚨 CRITICAL REGRESSION` status, the list of flipped test cases, and the non-zero exit code (`echo $?` or `$LASTEXITCODE` in PowerShell).*

### Running Unit & Integration Tests
```bash
pytest -v
```

---

## 3. Project Structure

```
Model Regression Detection System/
├── prompts/
│   ├── v1.yaml              # Baseline prompt (well-crafted classification & summary guidelines)
│   ├── v2.yaml              # Candidate prompt (improved sarcasm & edge case handling)
│   └── v3_regression.yaml   # Intentionally degraded prompt (demonstrates merge gating)
├── data/
│   └── golden_v1.json       # Hand-labeled golden dataset (55 curated test cases)
├── src/
│   ├── models.py            # Pydantic schemas (ClassificationResult, PromptConfig, GoldenCase, etc.)
│   ├── prompts.py           # YAML prompt loader
│   ├── classifier.py        # Feature under test (customer support email triage)
│   ├── judge.py             # LLM-as-a-Judge for factual summary evaluation (1-5 scoring)
│   ├── runner.py            # Async evaluation runner with Semaphore concurrency limiting
│   ├── store.py             # SQLite persistence for runs, metrics, and case results
│   ├── diff.py              # Comparison logic, regressions, improvements, thresholds, drift
│   ├── report.py            # Interactive HTML report (Chart.js) and GitHub PR summary markdown
│   ├── slack.py             # Slack Block Kit incoming webhook notifier
│   └── cli.py               # CLI entrypoint for CI/CD pipelines
├── tests/
│   ├── test_models.py       # Schema validation tests
│   ├── test_prompts.py      # Prompt config loading tests
│   ├── test_diff.py         # Diff logic, threshold boundaries, and drift detection tests
│   ├── test_store.py        # SQLite persistence and history retrieval tests
│   └── test_runner.py       # End-to-end evaluation & report generation tests
├── .github/workflows/
│   └── eval.yml             # GitHub Actions CI workflow (evaluates PRs on prompts/**)
├── Dockerfile               # Containerized runner for CI/CD runners & container jobs
├── thresholds.yaml          # Warning, critical, and rolling drift thresholds
├── requirements.txt         # Production dependencies
└── PLAN.md                  # Master architectural specification
```

---

## 4. How to Add Test Cases to the Golden Dataset

The golden dataset is stored in `data/golden_v1.json`. Each test case must be **hand-curated and human-verified**:

```json
{
  "id": "case_056",
  "email": "Can I set up annual wire transfer billing instead of monthly credit card charges?",
  "expected_category": "billing",
  "expected_summary": "Customer asks to pay via annual wire transfer instead of monthly credit card.",
  "difficulty": "medium",
  "notes": "Payment method policy edge case"
}
```

### Guidelines for New Cases:
1. **Never generate golden cases with an LLM**: Ground truth must reflect actual human expectations to avoid circular bias.
2. **Include Failure Cases**: Whenever a real user reports a production classification failure, sanitize the email and add it to the golden dataset as a regression test.
3. **Difficulty Tagging**: Mark cases as `easy`, `medium`, or `hard` (e.g., sarcasm, mixed language, or typos).

---

## 5. Tuning Thresholds (`thresholds.yaml`)

```yaml
# Maximum allowed drop in pass rate before triggering a warning (3%)
warn_threshold: 0.03

# Maximum allowed drop in pass rate before triggering a critical failure & blocking CI (8%)
critical_threshold: 0.08

# Rolling drift window configuration
drift_window: 7     # Number of historical runs to average
drift_floor: 0.90    # Alert if rolling average pass rate falls below 90%

# Concurrency & Performance
max_concurrency: 8
timeout_seconds: 30
```

> **Interview Talking Point:** In a dataset of 55 cases, one flipped case represents ~1.8% change. Setting a warning threshold at 3% (~2 cases) and a critical threshold at 8% (~4-5 cases) prevents noisy single-case fluctuations while strictly guarding against systemic regressions.

---

## 6. Key Architectural Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Temperature = 0.0** | Non-deterministic sampling introduces eval noise. For CI/CD regression testing, we need reproducible results where differences stem from prompt changes, not random seed jitter. |
| **Baseline Evaluated in Same Job** | CI runner environments (GitHub Actions, Docker) are ephemeral. Relying on remote or historical database state introduces race conditions. Running both or retrieving the keyed baseline ensures hermetic evaluation. |
| **Dual-Gating: Per-Run Diff + Rolling Drift** | Per-run diffs catch acute regressions. However, a series of -1.5% micro-regressions across 5 PRs would never trip a per-run threshold. The 7-run rolling average (`detect_drift`) catches gradual decay over time. |
| **LLM-as-a-Judge for Summaries** | Exact string matching fails on open-ended text. The judge rates semantic equivalence and penalizes invented/hallucinated details using a typed Pydantic schema (`JudgeScore`). |
| **Deterministic Mock Mode** | Allows unit tests and CI integration testing to run in seconds with zero external network dependencies and zero API costs. |

---

## 7. Containerization (Docker)

Build and run the evaluation container:
```bash
# Build Docker image
docker build -t model-regression-detector .

# Run with environment variables
docker run --rm \
  -e OPENAI_API_KEY="your-key" \
  -e SLACK_WEBHOOK_URL="your-webhook" \
  -v $(pwd)/report:/app/report \
  model-regression-detector --candidate v2 --baseline v1
```

---

## 8. CI/CD GitHub Actions Integration

The workflow in `.github/workflows/eval.yml`:
1. Triggers on pull requests modifying `prompts/**` or `data/**`.
2. Runs `python -m src.cli --candidate v2 --baseline v1 --out report`.
3. Posts the scorecard table directly as a comment on the PR.
4. Uploads the full interactive HTML diff report with Chart.js visualization as a build artifact.
5. **Fails the job with exit code 1 if critical regressions are found**, blocking the PR from merging when branch protection is enabled!
