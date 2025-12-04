# llmri/server.py

"""
FastAPI backend for llmri.

Current focus: serve results from feature_sweep.json so a frontend
can explore interventions.

Run with:

    export LLMRI_SWEEP_FILE="results/feature_sweep.json"   # optional
    uvicorn llmri.server:app --reload

Endpoints:

  - GET /health
      Simple liveness check.

  - GET /api/sweep/raw
      Return the raw JSON from the sweep file (model, runs, metrics).

  - GET /api/sweep/summary
      Compute lightweight summaries:
        * Top-N runs overall (by max |Δp|, then Σ|Δp|, then L2).
        * Top-M runs per feature.

      Query params:
        - top_overall: int = 10
        - top_per_feature: int = 5
        - feature: Optional[int] = None
        - prompt_substr: Optional[str] = None

  - GET /api/sweep/run/{idx}
      Return the full run (prompt, baseline_text, edited_text, metrics) by index.

This file deliberately does NOT yet talk to the live model. It’s for
visualizing the precomputed sweep results.
"""

from __future__ import annotations

import json
import math
import os
from fastapi.responses import HTMLResponse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config / sweep cache
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_PATH = Path("results/feature_sweep.json")


class SweepCache:
    """
    Simple file-based cache: reloads sweep JSON when the file changes.
    """

    def __init__(self) -> None:
        self._path: Optional[Path] = None
        self._mtime: Optional[float] = None
        self._data: Optional[Dict[str, Any]] = None

    def get_path(self) -> Path:
        env = os.getenv("LLMRI_SWEEP_FILE")
        if env:
            return Path(env)
        return DEFAULT_SWEEP_PATH

    def load(self) -> Dict[str, Any]:
        path = self.get_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Sweep file not found at {path!s}. "
                f"Set LLMRI_SWEEP_FILE or generate it with sweep_features.py."
            )

        mtime = path.stat().st_mtime
        if self._data is None or self._path != path or self._mtime != mtime:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._path = path
            self._mtime = mtime
            self._data = data

        assert self._data is not None
        return self._data


SWEEP_CACHE = SweepCache()


# ---------------------------------------------------------------------------
# Internal summary representation (mirrors analyze_feature_sweep.py)
# ---------------------------------------------------------------------------

@dataclass
class RunSummaryInternal:
    idx: int
    prompt: str
    feature_id: int
    scale: float
    l2_diff: float
    cosine_similarity: float
    changed_top1: bool
    max_abs_delta_p: float
    sum_abs_delta_p: float


def _safe_log(x: float) -> float:
    eps = 1e-12
    return math.log(max(x, eps))


def _summarize_run(run: Dict[str, Any], idx: int) -> RunSummaryInternal:
    prompt = run["prompt"]
    feature_id = int(run["feature_id"])
    scale = float(run["scale"])

    m = run["metrics"]
    l2_diff = float(m.get("l2_diff", 0.0))
    cosine_similarity = float(m.get("cosine_similarity", 1.0))
    changed_top1 = bool(m.get("top_token_changed", False))

    topk = m.get("topk_deltas", []) or []

    max_abs_delta_p = 0.0
    sum_abs_delta_p = 0.0

    for td in topk:
        p_base = float(td.get("p_base", 0.0))
        p_edit = float(td.get("p_edit", 0.0))
        delta = float(td.get("delta", p_edit - p_base))

        abs_d = abs(delta)
        if abs_d > max_abs_delta_p:
            max_abs_delta_p = abs_d
        sum_abs_delta_p += abs_d

    return RunSummaryInternal(
        idx=idx,
        prompt=prompt,
        feature_id=feature_id,
        scale=scale,
        l2_diff=l2_diff,
        cosine_similarity=cosine_similarity,
        changed_top1=changed_top1,
        max_abs_delta_p=max_abs_delta_p,
        sum_abs_delta_p=sum_abs_delta_p,
    )


def _load_summaries(
    only_feature: Optional[int],
    prompt_substr: Optional[str],
) -> List[RunSummaryInternal]:
    data = SWEEP_CACHE.load()
    runs = data.get("runs", [])
    out: List[RunSummaryInternal] = []

    for i, r in enumerate(runs):
        rs = _summarize_run(r, idx=i)

        if only_feature is not None and rs.feature_id != only_feature:
            continue

        if prompt_substr is not None:
            if prompt_substr.lower() not in rs.prompt.lower():
                continue

        out.append(rs)

    return out


# ---------------------------------------------------------------------------
# Pydantic models for API responses
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    idx: int
    feature_id: int
    scale: float
    prompt: str
    l2_diff: float
    cosine_similarity: float
    changed_top1: bool
    max_abs_delta_p: float
    sum_abs_delta_p: float


class SweepSummaryResponse(BaseModel):
    meta: Dict[str, Any]
    overall: List[RunSummary]
    per_feature: Dict[str, List[RunSummary]]


class SweepRunResponse(BaseModel):
    idx: int
    prompt: str
    feature_id: int
    scale: float
    baseline_text: str
    edited_text: str
    metrics: Dict[str, Any]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="llmri backend", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sweep/raw")
def api_sweep_raw() -> Dict[str, Any]:
    """
    Return the raw sweep JSON (model + all runs).
    """
    try:
        data = SWEEP_CACHE.load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return data


@app.get("/api/sweep/summary", response_model=SweepSummaryResponse)
def api_sweep_summary(
    top_overall: int = Query(10, ge=1, le=100),
    top_per_feature: int = Query(5, ge=1, le=50),
    feature: Optional[int] = Query(None),
    prompt_substr: Optional[str] = Query(None),
) -> SweepSummaryResponse:
    """
    Summarize the sweep:

      - Top-N runs overall (by max |Δp|, then Σ|Δp|, then L2).
      - Top-M runs per feature.

    Optional filters:
      - feature: restrict to a single feature id.
      - prompt_substr: restrict to prompts containing this substring.
    """
    try:
        data = SWEEP_CACHE.load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    meta = {
        k: data.get(k)
        for k in ("model", "device", "sae_checkpoint", "layer_index", "feature_ids", "scales", "prompts")
        if k in data
    }

    summaries = _load_summaries(only_feature=feature, prompt_substr=prompt_substr)
    if not summaries:
        return SweepSummaryResponse(meta=meta, overall=[], per_feature={})

    # Overall: sort and slice
    sorted_overall = sorted(
        summaries,
        key=lambda r: (r.max_abs_delta_p, r.sum_abs_delta_p, r.l2_diff),
        reverse=True,
    )
    overall = [
        RunSummary(
            idx=r.idx,
            feature_id=r.feature_id,
            scale=r.scale,
            prompt=r.prompt,
            l2_diff=r.l2_diff,
            cosine_similarity=r.cosine_similarity,
            changed_top1=r.changed_top1,
            max_abs_delta_p=r.max_abs_delta_p,
            sum_abs_delta_p=r.sum_abs_delta_p,
        )
        for r in sorted_overall[:top_overall]
    ]

    # Per-feature: group and slice
    per_feat: Dict[int, List[RunSummaryInternal]] = {}
    for r in summaries:
        per_feat.setdefault(r.feature_id, []).append(r)

    per_feature_out: Dict[str, List[RunSummary]] = {}
    for feat_id, rs in sorted(per_feat.items()):
        rs_sorted = sorted(
            rs,
            key=lambda r: (r.max_abs_delta_p, r.sum_abs_delta_p, r.l2_diff),
            reverse=True,
        )
        per_feature_out[str(feat_id)] = [
            RunSummary(
                idx=r.idx,
                feature_id=r.feature_id,
                scale=r.scale,
                prompt=r.prompt,
                l2_diff=r.l2_diff,
                cosine_similarity=r.cosine_similarity,
                changed_top1=r.changed_top1,
                max_abs_delta_p=r.max_abs_delta_p,
                sum_abs_delta_p=r.sum_abs_delta_p,
            )
            for r in rs_sorted[:top_per_feature]
        ]

    return SweepSummaryResponse(meta=meta, overall=overall, per_feature=per_feature_out)


@app.get("/api/sweep/run/{idx}", response_model=SweepRunResponse)
def api_sweep_run(idx: int) -> SweepRunResponse:
    """
    Return the full run (baseline/edited text + metrics) by index.
    """
    try:
        data = SWEEP_CACHE.load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    runs = data.get("runs", [])
    if idx < 0 or idx >= len(runs):
        raise HTTPException(
            status_code=404,
            detail=f"Run index {idx} out of range (0..{len(runs) - 1}).",
        )

    r = runs[idx]
    m = r.get("metrics", {})

    return SweepRunResponse(
        idx=idx,
        prompt=r.get("prompt", ""),
        feature_id=int(r.get("feature_id", -1)),
        scale=float(r.get("scale", 1.0)),
        baseline_text=r.get("baseline_text", ""),
        edited_text=r.get("edited_text", ""),
        metrics=m,
    )








@app.get("/", response_class=HTMLResponse)
async def sweep_viewer() -> HTMLResponse:
    # Simple static HTML/JS UI. Served from the same origin so no CORS issues.
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>LLMRI – Feature Sweep Viewer</title>
  <style>
    :root {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color-scheme: light dark;
    }
    body {
      margin: 0;
      padding: 0 1.5rem 2rem;
      background: #0b0c10;
      color: #e5e7eb;
    }
    h1, h2, h3 {
      margin-top: 1.2rem;
      margin-bottom: 0.4rem;
    }
    .meta {
      margin-top: 1rem;
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      background: #111827;
      font-size: 0.9rem;
      line-height: 1.4;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 0.5rem 1.5rem;
    }
    .meta span.label {
      font-weight: 600;
      color: #9ca3af;
      margin-right: 0.25rem;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin-top: 0.5rem;
      font-size: 0.85rem;
    }
    thead {
      background: #111827;
    }
    th, td {
      border: 1px solid #1f2933;
      padding: 0.3rem 0.45rem;
      text-align: left;
      white-space: nowrap;
    }
    th {
      font-weight: 600;
      color: #d1d5db;
    }
    tbody tr:nth-child(even) {
      background: #020617;
    }
    tbody tr:nth-child(odd) {
      background: #030712;
    }
    tbody tr:hover {
      background: #1e293b;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.8rem;
    }
    .prompt-cell {
      max-width: 320px;
      white-space: normal;
    }
    .badge {
      display: inline-block;
      padding: 0.05rem 0.35rem;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge-yes {
      background: #16a34a33;
      color: #bbf7d0;
      border: 1px solid #16a34a;
    }
    .badge-no {
      background: #b91c1c33;
      color: #fecaca;
      border: 1px solid #b91c1c;
    }
    details {
      margin-top: 0.75rem;
      border-radius: 0.5rem;
      background: #020617;
      border: 1px solid #111827;
      padding: 0.4rem 0.7rem 0.6rem;
    }
    details > summary {
      cursor: pointer;
      list-style: none;
      font-weight: 600;
      color: #e5e7eb;
    }
    details > summary::-webkit-details-marker {
      display: none;
    }
    .summary-row {
      display: inline-flex;
      gap: 0.75rem;
      align-items: center;
      font-size: 0.8rem;
      color: #9ca3af;
      margin-left: 0.3rem;
    }
    .pill {
      padding: 0.05rem 0.4rem;
      border-radius: 999px;
      background: #111827;
      border: 1px solid #1f2933;
    }
    .error {
      margin-top: 1rem;
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      background: #450a0a;
      color: #fecaca;
      font-size: 0.9rem;
    }
    .loading {
      margin-top: 1rem;
      font-size: 0.9rem;
      color: #9ca3af;
    }
  </style>
</head>
<body>
  <h1>LLMRI – Feature Sweep Viewer</h1>
  <div id="status" class="loading">Loading summary from <code>/api/sweep/summary</code>…</div>

  <section id="meta"></section>

  <section id="overall">
    <h2>Top overall runs</h2>
    <div id="overall-table"></div>
  </section>

  <section id="per-feature">
    <h2>Per-feature breakdown</h2>
    <div id="per-feature-list"></div>
  </section>

  <script>
    function fmtPct(x) {
      return (x * 100).toFixed(1) + "%";
    }
    function fmtNum(x, digits = 3) {
      return Number(x).toFixed(digits);
    }

    function renderMeta(meta) {
      const el = document.getElementById("meta");
      const featIds = (meta.feature_ids || []).join(", ");
      const scales = (meta.scales || []).join(", ");
      const prompts = (meta.prompts || []).join(" | ");

      el.innerHTML = `
        <div class="meta">
          <div><span class="label">Model:</span><code>${meta.model}</code></div>
          <div><span class="label">Device:</span><code>${meta.device}</code></div>
          <div><span class="label">Layer index:</span><code>${meta.layer_index}</code></div>
          <div><span class="label">SAE checkpoint:</span><code>${meta.sae_checkpoint}</code></div>
          <div><span class="label">Feature IDs:</span><code>${featIds}</code></div>
          <div><span class="label">Scales:</span><code>${scales}</code></div>
          <div style="grid-column: 1 / -1;"><span class="label">Prompts:</span><code>${prompts}</code></div>
        </div>
      `;
    }

    function makeTable(rows, opts = {}) {
      const cols = opts.columns;
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");

      const headerRow = document.createElement("tr");
      cols.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col.label;
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);

      rows.forEach(r => {
        const tr = document.createElement("tr");
        cols.forEach(col => {
          const td = document.createElement("td");
          const v = col.render ? col.render(r) : r[col.key];
          if (col.className) td.className = col.className;
          if (v instanceof Node) td.appendChild(v); else td.textContent = v;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);
      return table;
    }

    function renderOverall(overall) {
      const container = document.getElementById("overall-table");
      if (!overall || overall.length === 0) {
        container.textContent = "No runs found.";
        return;
      }

      const rows = overall.slice(0, 50); // cap for sanity
      const cols = [
        { key: "idx", label: "#" },
        { key: "feature_id", label: "Feature" },
        { key: "scale", label: "Scale" },
        {
          key: "max_abs_delta_p",
          label: "max |Δp|",
          render: r => fmtPct(r.max_abs_delta_p),
        },
        {
          key: "sum_abs_delta_p",
          label: "Σ |Δp|",
          render: r => fmtPct(r.sum_abs_delta_p),
        },
        {
          key: "changed_top1",
          label: "Δ top-1",
          render: r => {
            const span = document.createElement("span");
            span.className = "badge " + (r.changed_top1 ? "badge-yes" : "badge-no");
            span.textContent = r.changed_top1 ? "changed" : "same";
            return span;
          },
        },
        {
          key: "l2_diff",
          label: "L2",
          render: r => fmtNum(r.l2_diff, 2),
        },
        {
          key: "cosine_similarity",
          label: "cos",
          render: r => fmtNum(r.cosine_similarity, 4),
        },
        {
          key: "prompt",
          label: "Prompt",
          className: "prompt-cell",
        },
      ];

      container.innerHTML = "";
      container.appendChild(makeTable(rows, { columns: cols }));
    }

    function renderPerFeature(perFeature) {
      const container = document.getElementById("per-feature-list");
      container.innerHTML = "";

      const entries = Object.entries(perFeature || {});
      if (entries.length === 0) {
        container.textContent = "No per-feature data.";
        return;
      }

      // Sort by feature id numeric
      entries.sort((a, b) => Number(a[0]) - Number(b[0]));

      for (const [featId, runs] of entries) {
        const details = document.createElement("details");
        details.open = false;

        const summary = document.createElement("summary");
        summary.innerHTML = `
          Feature <code>${featId}</code>
          <span class="summary-row">
            <span class="pill">runs: ${runs.length}</span>
          </span>
        `;
        details.appendChild(summary);

        const cols = [
          { key: "idx", label: "#" },
          { key: "scale", label: "Scale" },
          {
            key: "max_abs_delta_p",
            label: "max |Δp|",
            render: r => fmtPct(r.max_abs_delta_p),
          },
          {
            key: "sum_abs_delta_p",
            label: "Σ |Δp|",
            render: r => fmtPct(r.sum_abs_delta_p),
          },
          {
            key: "changed_top1",
            label: "Δ top-1",
            render: r => {
              const span = document.createElement("span");
              span.className = "badge " + (r.changed_top1 ? "badge-yes" : "badge-no");
              span.textContent = r.changed_top1 ? "changed" : "same";
              return span;
            },
          },
          {
            key: "l2_diff",
            label: "L2",
            render: r => fmtNum(r.l2_diff, 2),
          },
          {
            key: "cosine_similarity",
            label: "cos",
            render: r => fmtNum(r.cosine_similarity, 4),
          },
          {
            key: "prompt",
            label: "Prompt",
            className: "prompt-cell",
          },
        ];

        // Default: sort runs for this feature by max_abs_delta_p desc
        const sortedRuns = runs.slice().sort(
          (a, b) => b.max_abs_delta_p - a.max_abs_delta_p
        );

        details.appendChild(makeTable(sortedRuns, { columns: cols }));
        container.appendChild(details);
      }
    }

    async function main() {
      const status = document.getElementById("status");
      try {
        const res = await fetch("/api/sweep/summary");
        if (!res.ok) {
          throw new Error("HTTP " + res.status);
        }
        const data = await res.json();
        status.textContent = "";

        renderMeta(data.meta || {});
        renderOverall(data.overall || []);
        renderPerFeature(data.per_feature || {});
      } catch (err) {
        console.error(err);
        status.className = "error";
        status.textContent = "Error loading /api/sweep/summary: " + err;
      }
    }

    main();
  </script>
</body>
</html>
        """
    )