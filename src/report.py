from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import Template

from src.models import (
    CaseEvalResult,
    DiffResult,
    GoldenCase,
    RunSummary,
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Regression Diff Report | {{ diff.candidate_version }} vs {{ diff.baseline_version }}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --accent: #58a6ff;
      --pass: #238636;
      --pass-bg: rgba(35, 134, 54, 0.15);
      --warn: #d29922;
      --warn-bg: rgba(210, 153, 34, 0.15);
      --crit: #da3633;
      --crit-bg: rgba(218, 54, 51, 0.15);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 2rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }
    h1 { color: var(--text-bright); font-size: 1.8rem; }
    .badge {
      display: inline-block;
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-pass { background-color: var(--pass-bg); color: #3fb950; border: 1px solid #238636; }
    .badge-warning { background-color: var(--warn-bg); color: #e3b341; border: 1px solid #9e6a03; }
    .badge-critical { background-color: var(--crit-bg); color: #f85149; border: 1px solid #da3633; }

    .grid-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
    }
    .card-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; margin-bottom: 0.5rem; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: var(--text-bright); }
    .delta-pos { color: #3fb950; font-size: 1rem; margin-left: 0.3rem; }
    .delta-neg { color: #f85149; font-size: 1rem; margin-left: 0.3rem; }

    .chart-section {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }
    .section-title {
      font-size: 1.25rem;
      color: var(--text-bright);
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      font-size: 0.9rem;
    }
    th, td {
      padding: 0.85rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: #21262d;
      color: #8b949e;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }
    tr:hover { background-color: rgba(255, 255, 255, 0.02); }

    .tag {
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .tag-easy { background: rgba(56, 139, 253, 0.15); color: #58a6ff; }
    .tag-medium { background: rgba(210, 153, 34, 0.15); color: #d29922; }
    .tag-hard { background: rgba(218, 54, 51, 0.15); color: #f85149; }

    .pill-fail { color: #f85149; font-weight: 600; }
    .pill-pass { color: #3fb950; font-weight: 600; }
    .meta-sub { font-size: 0.8rem; color: #8b949e; margin-top: 0.25rem; }
    .alert-banner {
      padding: 1rem;
      border-radius: 6px;
      margin-bottom: 1.5rem;
      font-weight: 500;
    }
    .alert-banner-crit { background: var(--crit-bg); border: 1px solid var(--crit); color: #f85149; }
    .alert-banner-warn { background: var(--warn-bg); border: 1px solid var(--warn); color: #e3b341; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>LLM Regression Diff Report</h1>
        <div class="meta-sub">
          Candidate: <strong>{{ diff.candidate_version }}</strong> vs Baseline: <strong>{{ diff.baseline_version }}</strong>
          | Model: {{ candidate_summary.model }} | Evaluated at: {{ candidate_summary.created_at }}
        </div>
      </div>
      <div>
        <span class="badge badge-{{ diff.status }}">{{ diff.status }}</span>
      </div>
    </header>

    {% if diff.drift_detected %}
    <div class="alert-banner alert-banner-warn">
      ⚠️ <strong>Drift Warning:</strong> {{ diff.drift_message }}
    </div>
    {% endif %}

    {% if diff.status == 'critical' %}
    <div class="alert-banner alert-banner-crit">
      🚨 <strong>Critical Regression Detected:</strong> Pass rate dropped by {{ (diff.delta * -100) | round(1) }}%. Merge is blocked!
    </div>
    {% endif %}

    <div class="grid-cards">
      <div class="card">
        <div class="card-title">Pass Rate</div>
        <div class="card-value">
          {{ (diff.candidate_pass_rate * 100) | round(1) }}%
          {% if diff.delta >= 0 %}
          <span class="delta-pos">+{{ (diff.delta * 100) | round(1) }}%</span>
          {% else %}
          <span class="delta-neg">{{ (diff.delta * 100) | round(1) }}%</span>
          {% endif %}
        </div>
        <div class="meta-sub">Baseline: {{ (diff.baseline_pass_rate * 100) | round(1) }}%</div>
      </div>

      <div class="card">
        <div class="card-title">LLM Judge Avg (1-5)</div>
        <div class="card-value">
          {{ candidate_summary.avg_judge }} / 5.0
        </div>
        <div class="meta-sub">Baseline: {{ baseline_summary.avg_judge }} / 5.0</div>
      </div>

      <div class="card">
        <div class="card-title">Regressed Cases</div>
        <div class="card-value" style="color: {% if diff.regressions | length > 0 %}#f85149{% else %}#3fb950{% endif %};">
          {{ diff.regressions | length }}
        </div>
        <div class="meta-sub">{{ diff.improvements | length }} improvements detected</div>
      </div>

      <div class="card">
        <div class="card-title">Avg Latency & Tokens</div>
        <div class="card-value">{{ candidate_summary.avg_latency_ms }} ms</div>
        <div class="meta-sub">{{ candidate_summary.total_tokens }} tokens consumed</div>
      </div>
    </div>

    <!-- Charts -->
    <div class="chart-section">
      <div class="section-title">📊 Per-Category Accuracy & Historical Trend</div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
        <div>
          <canvas id="categoryChart" height="200"></canvas>
        </div>
        <div>
          <canvas id="trendChart" height="200"></canvas>
        </div>
      </div>
    </div>

    <!-- Regressions Section -->
    <div class="chart-section">
      <div class="section-title">🚨 Regressions (Flipped from Pass to Fail: {{ diff.regressions | length }})</div>
      {% if diff.regressions | length == 0 %}
      <p style="color: #3fb950; font-weight: 500;">No regressions detected! All baseline passing cases continue to pass.</p>
      {% else %}
      <table>
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Difficulty</th>
            <th>Expected</th>
            <th>Baseline (Old)</th>
            <th>Candidate (New)</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {% for r in regressed_details %}
          <tr>
            <td><strong>{{ r.case.id }}</strong></td>
            <td><span class="tag tag-{{ r.case.difficulty }}">{{ r.case.difficulty }}</span></td>
            <td>
              <div><strong>{{ r.case.expected_category }}</strong></div>
              <div class="meta-sub">{{ r.case.expected_summary }}</div>
            </td>
            <td>
              <div class="pill-pass">✓ {{ r.base.pred_category }}</div>
              <div class="meta-sub">{{ r.base.pred_summary }}</div>
            </td>
            <td>
              <div class="pill-fail">✗ {{ r.cand.pred_category }}</div>
              <div class="meta-sub">{{ r.cand.pred_summary }}</div>
            </td>
            <td class="meta-sub">{{ r.case.notes }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>

    <!-- Improvements Section -->
    <div class="chart-section">
      <div class="section-title">✨ Improvements (Flipped from Fail to Pass: {{ diff.improvements | length }})</div>
      {% if diff.improvements | length == 0 %}
      <p style="color: #8b949e;">No new test cases flipped to pass in this run.</p>
      {% else %}
      <table>
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Difficulty</th>
            <th>Expected</th>
            <th>Baseline (Old)</th>
            <th>Candidate (New)</th>
          </tr>
        </thead>
        <tbody>
          {% for r in improved_details %}
          <tr>
            <td><strong>{{ r.case.id }}</strong></td>
            <td><span class="tag tag-{{ r.case.difficulty }}">{{ r.case.difficulty }}</span></td>
            <td>
              <div><strong>{{ r.case.expected_category }}</strong></div>
              <div class="meta-sub">{{ r.case.expected_summary }}</div>
            </td>
            <td>
              <div class="pill-fail">✗ {{ r.base.pred_category }}</div>
            </td>
            <td>
              <div class="pill-pass">✓ {{ r.cand.pred_category }}</div>
              <div class="meta-sub">{{ r.cand.pred_summary }}</div>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>
  </div>

  <script>
    // Category Chart
    const catLabels = {{ category_labels | tojson }};
    const catDeltas = {{ category_deltas | tojson }};
    new Chart(document.getElementById('categoryChart'), {
      type: 'bar',
      data: {
        labels: catLabels,
        datasets: [{
          label: 'Category Accuracy Delta',
          data: catDeltas,
          backgroundColor: catDeltas.map(v => v >= 0 ? 'rgba(63, 185, 80, 0.7)' : 'rgba(248, 81, 73, 0.7)'),
          borderColor: catDeltas.map(v => v >= 0 ? '#238636' : '#da3633'),
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Category Delta Breakdown', color: '#c9d1d9' }
        },
        scales: {
          y: {
            grid: { color: '#30363d' },
            ticks: { color: '#8b949e', callback: v => (v * 100).toFixed(0) + '%' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#8b949e' }
          }
        }
      }
    });

    // Trend Chart
    const trendLabels = {{ trend_labels | tojson }};
    const trendPassRates = {{ trend_pass_rates | tojson }};
    new Chart(document.getElementById('trendChart'), {
      type: 'line',
      data: {
        labels: trendLabels,
        datasets: [{
          label: 'Pass Rate History',
          data: trendPassRates,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Pass Rate Over Recent Runs', color: '#c9d1d9' }
        },
        scales: {
          y: {
            min: 0.5,
            max: 1.0,
            grid: { color: '#30363d' },
            ticks: { color: '#8b949e', callback: v => (v * 100).toFixed(0) + '%' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#8b949e' }
          }
        }
      }
    });
  </script>
</body>
</html>
"""


def generate_html_report(
    diff: DiffResult,
    baseline_summary: RunSummary,
    candidate_summary: RunSummary,
    baseline_results: List[CaseEvalResult],
    candidate_results: List[CaseEvalResult],
    cases: List[GoldenCase],
    history_runs: Optional[List[RunSummary]] = None,
    out_dir: str | Path = "report"
) -> Path:
    """Generate an interactive HTML diff report with Chart.js visualization."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "diff_report.html"

    case_map = {c.id: c for c in cases}
    base_map = {r.case_id: r for r in baseline_results}
    cand_map = {r.case_id: r for r in candidate_results}

    regressed_details = [
        {
            "case": case_map[cid],
            "base": base_map[cid],
            "cand": cand_map[cid],
        }
        for cid in diff.regressions
        if cid in case_map and cid in base_map and cid in cand_map
    ]

    improved_details = [
        {
            "case": case_map[cid],
            "base": base_map[cid],
            "cand": cand_map[cid],
        }
        for cid in diff.improvements
        if cid in case_map and cid in base_map and cid in cand_map
    ]

    cat_labels = list(diff.per_category.keys())
    cat_deltas = [diff.per_category[k] for k in cat_labels]

    if history_runs and len(history_runs) > 0:
        trend_labels = [f"Run {r.prompt_version}" for r in history_runs]
        trend_pass_rates = [r.pass_rate for r in history_runs]
    else:
        trend_labels = [f"Baseline ({baseline_summary.prompt_version})", f"Candidate ({candidate_summary.prompt_version})"]
        trend_pass_rates = [baseline_summary.pass_rate, candidate_summary.pass_rate]

    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        diff=diff,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        regressed_details=regressed_details,
        improved_details=improved_details,
        category_labels=cat_labels,
        category_deltas=cat_deltas,
        trend_labels=trend_labels,
        trend_pass_rates=trend_pass_rates,
    )

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_file


def generate_pr_summary_markdown(
    diff: DiffResult,
    baseline_summary: RunSummary,
    candidate_summary: RunSummary,
    out_dir: str | Path = "report"
) -> Path:
    """Generate concise GitHub PR comment markdown (summary.md)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_file = out_path / "summary.md"

    emoji = {"pass": "✅", "warning": "⚠️", "critical": "🚨"}[diff.status]
    delta_str = f"+{(diff.delta * 100):.1f}%" if diff.delta >= 0 else f"{(diff.delta * 100):.1f}%"

    md = [
        f"### {emoji} Model Regression Eval: **{diff.status.upper()}**",
        "",
        f"| Metric | Baseline (`{diff.baseline_version}`) | Candidate (`{diff.candidate_version}`) | Delta |",
        "|---|---|---|---|",
        f"| **Pass Rate** | {baseline_summary.pass_rate * 100:.1f}% | {candidate_summary.pass_rate * 100:.1f}% | **{delta_str}** |",
        f"| **Judge Score (1-5)** | {baseline_summary.avg_judge:.2f} | {candidate_summary.avg_judge:.2f} | {(candidate_summary.avg_judge - baseline_summary.avg_judge):+.2f} |",
        f"| **Latency (avg)** | {baseline_summary.avg_latency_ms:.0f} ms | {candidate_summary.avg_latency_ms:.0f} ms | {(candidate_summary.avg_latency_ms - baseline_summary.avg_latency_ms):+.0f} ms |",
        f"| **Regressions** | - | - | **{len(diff.regressions)}** flipped to fail |",
        f"| **Improvements** | - | - | **{len(diff.improvements)}** flipped to pass |",
        "",
    ]

    if diff.drift_detected:
        md.append(f"> ⚠️ **Drift Alert:** {diff.drift_message}\n")

    if diff.status == "critical":
        md.append("> 🚨 **MERGE BLOCKED:** Critical regression threshold exceeded. Please review regressed cases before merging.\n")

    if diff.regressions:
        md.append("#### Regressed Cases:")
        for r_id in diff.regressions[:5]:
            md.append(f"- Case `{r_id}` flipped from PASS to FAIL")
        if len(diff.regressions) > 5:
            md.append(f"- *...and {len(diff.regressions) - 5} more cases*")
        md.append("")

    md.append("*(Full HTML visual diff report is attached to workflow artifacts)*")

    content = "\n".join(md)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(content)

    return summary_file
