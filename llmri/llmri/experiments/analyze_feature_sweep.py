"""
Analyze feature_sweep.json and surface the strongest interventions.

Usage (from repo root):

  python -m llmri.experiments.analyze_feature_sweep \
    --file results/feature_sweep.json \
    --top-overall 10 \
    --top-per-feature 5

The input is expected to be the JSON produced by sweep_features.py, e.g.:

{
  "model": "...",
  "device": "...",
  "sae_checkpoint": "...",
  "layer_index": 10,
  "feature_ids": [...],
  "scales": [...],
  "prompts": [...],
  "runs": [
    {
      "prompt": "It matters much that you should",
      "feature_id": 393,
      "scale": 6.0,
      "baseline_text": "...",
      "edited_text": "...",
      "metrics": {
        "l2_diff": ...,
        "cosine_similarity": ...,
        "top_token_id_base": ...,
        "top_token_id_edit": ...,
        "top_token_changed": true/false,
        "topk_deltas": [
          {
            "token_id": ...,
            "token_str": "...",
            "p_base": ...,
            "p_edit": ...,
            "delta": ...
          },
          ...
        ]
      }
    },
    ...
  ]
}

This script computes simple scalar â€œeffect sizeâ€ metrics for each run and prints
ranked summaries:

  * Top N strongest runs overall.
  * For each feature, its top M runs.

Metrics we compute per run:

  - max_abs_delta_p   : max_i |Î”p_i| over the top-k tokens logged.
  - sum_abs_delta_p   : sum_i |Î”p_i|.
  - changed_top1      : 1 if top_token_changed else 0.
  - l2_diff           : directly from the file.
  - local_KL          : KL(p_edit || p_base) over the *logged* tokens only
                        (not a full-vocab KL, just a local proxy).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ----------------------------- data structures -----------------------------


@dataclass
class RunSummary:
    idx: int
    prompt: str
    feature_id: int
    scale: float
    l2_diff: float
    cosine_similarity: float
    changed_top1: bool
    max_abs_delta_p: float
    sum_abs_delta_p: float
    local_kl: float  # local KL(p_edit || p_base) over logged tokens
    baseline_text: str
    edited_text: str


# ----------------------------- helpers -------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze feature_sweep.json results.")
    p.add_argument(
        "--file",
        required=True,
        help="Path to feature_sweep.json produced by sweep_features.py.",
    )
    p.add_argument(
        "--top-overall",
        type=int,
        default=10,
        help="How many top runs overall to show (by max_abs_delta_p).",
    )
    p.add_argument(
        "--top-per-feature",
        type=int,
        default=5,
        help="How many top runs per feature to show.",
    )
    p.add_argument(
        "--feature",
        type=int,
        default=None,
        help="Optional: restrict analysis to a single feature id.",
    )
    p.add_argument(
        "--prompt-substr",
        type=str,
        default=None,
        help="Optional: restrict runs to prompts containing this substring.",
    )
    return p.parse_args()


def safe_log(x: float) -> float:
    # Avoid log(0); clamp to a tiny epsilon.
    eps = 1e-12
    return math.log(max(x, eps))


def summarize_run(run: Dict[str, Any], idx: int) -> RunSummary:
    prompt = run["prompt"]
    feature_id = int(run["feature_id"])
    scale = float(run["scale"])
    baseline_text = run.get("baseline_text", "")
    edited_text = run.get("edited_text", "")

    m = run["metrics"]
    l2_diff = float(m.get("l2_diff", 0.0))
    cosine_similarity = float(m.get("cosine_similarity", 1.0))
    changed_top1 = bool(m.get("top_token_changed", False))

    topk = m.get("topk_deltas", []) or []

    max_abs_delta_p = 0.0
    sum_abs_delta_p = 0.0
    local_kl = 0.0

    for td in topk:
        p_base = float(td.get("p_base", 0.0))
        p_edit = float(td.get("p_edit", 0.0))
        delta = float(td.get("delta", p_edit - p_base))

        abs_d = abs(delta)
        if abs_d > max_abs_delta_p:
            max_abs_delta_p = abs_d
        sum_abs_delta_p += abs_d

        # local KL(p_edit || p_base) over the logged tokens only
        if p_edit > 0.0 and p_base > 0.0:
            local_kl += p_edit * (safe_log(p_edit) - safe_log(p_base))

    return RunSummary(
        idx=idx,
        prompt=prompt,
        feature_id=feature_id,
        scale=scale,
        l2_diff=l2_diff,
        cosine_similarity=cosine_similarity,
        changed_top1=changed_top1,
        max_abs_delta_p=max_abs_delta_p,
        sum_abs_delta_p=sum_abs_delta_p,
        local_kl=local_kl,
        baseline_text=baseline_text,
        edited_text=edited_text,
    )


def load_summaries(
    path: str,
    only_feature: Optional[int],
    prompt_substr: Optional[str],
) -> List[RunSummary]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data["runs"]
    out: List[RunSummary] = []

    for i, r in enumerate(runs):
        rs = summarize_run(r, idx=i)

        if only_feature is not None and rs.feature_id != only_feature:
            continue

        if prompt_substr is not None:
            if prompt_substr.lower() not in rs.prompt.lower():
                continue

        out.append(rs)

    return out


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def fmt_bool(b: bool) -> str:
    return "Y" if b else " "


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print()
    print(bar)
    print(title)
    print(bar)


def short_text(s: str, max_len: int = 80) -> str:
    s2 = s.replace("\n", " ")
    return s2 if len(s2) <= max_len else s2[: max_len - 3] + "..."


# ----------------------------- printing ------------------------------------


def print_overall_top(runs: List[RunSummary], k: int) -> None:
    if not runs:
        print("\n[overall] No runs after filtering.")
        return

    sorted_runs = sorted(
        runs,
        key=lambda r: (r.max_abs_delta_p, r.sum_abs_delta_p, r.l2_diff),
        reverse=True,
    )

    print_header(f"Top {min(k, len(sorted_runs))} runs overall (by max |Î”p|)")
    print(
        f"{'idx':>4}  {'feat':>5}  {'scale':>5}  {'max|Î”p|':>8}  "
        f"{'Î£|Î”p|':>8}  {'Î”top1':>5}  {'L2':>8}  {'cos':>7}  prompt"
    )
    for r in sorted_runs[:k]:
        print(
            f"{r.idx:4d}  {r.feature_id:5d}  {r.scale:5.1f}  "
            f"{fmt_pct(r.max_abs_delta_p):>8}  {fmt_pct(r.sum_abs_delta_p):>8}  "
            f"{fmt_bool(r.changed_top1):>5}  {r.l2_diff:8.2f}  {r.cosine_similarity:7.4f}  "
            f"{short_text(r.prompt)}"
        )


def print_top_per_feature(runs: List[RunSummary], k: int) -> None:
    if not runs:
        print("\n[per-feature] No runs after filtering.")
        return

    runs_by_feat: Dict[int, List[RunSummary]] = {}
    for r in runs:
        runs_by_feat.setdefault(r.feature_id, []).append(r)

    for feat_id, rs in sorted(runs_by_feat.items()):
        sorted_rs = sorted(
            rs,
            key=lambda r: (r.max_abs_delta_p, r.sum_abs_delta_p, r.l2_diff),
            reverse=True,
        )
        print_header(f"Feature {feat_id}: top {min(k, len(sorted_rs))} runs")
        print(
            f"{'idx':>4}  {'scale':>5}  {'max|Î”p|':>8}  {'Î£|Î”p|':>8}  "
            f"{'Î”top1':>5}  {'L2':>8}  {'cos':>7}  prompt"
        )
        for r in sorted_rs[:k]:
            print(
                f"{r.idx:4d}  {r.scale:5.1f}  "
                f"{fmt_pct(r.max_abs_delta_p):>8}  {fmt_pct(r.sum_abs_delta_p):>8}  "
                f"{fmt_bool(r.changed_top1):>5}  {r.l2_diff:8.2f}  {r.cosine_similarity:7.4f}  "
                f"{short_text(r.prompt)}"
            )


def print_example_snippets(runs: List[RunSummary], k: int) -> None:
    if not runs:
        return

    sorted_runs = sorted(
        runs,
        key=lambda r: (r.max_abs_delta_p, r.sum_abs_delta_p, r.l2_diff),
        reverse=True,
    )
    top = sorted_runs[:k]

    print_header(f"Example baseline vs edited text for top {len(top)} runs")
    for r in top:
        print(f"\n--- idx={r.idx} feat={r.feature_id} scale={r.scale} ---")
        print(f"prompt: {r.prompt}")
        print(f"max|Î”p|={fmt_pct(r.max_abs_delta_p)}, Î£|Î”p|={fmt_pct(r.sum_abs_delta_p)}, "
              f"L2={r.l2_diff:.2f}, cos={r.cosine_similarity:.4f}, Î”top1={r.changed_top1}")
        print("\n[baseline]")
        print(short_text(r.baseline_text, max_len=400))
        print("\n[edited]")
        print(short_text(r.edited_text, max_len=400))


# ----------------------------- main ----------------------------------------


def main() -> None:
    args = parse_args()

    runs = load_summaries(
        path=args.file,
        only_feature=args.feature,
        prompt_substr=args.prompt_substr,
    )

    if not runs:
        print(f"No runs found in {args.file!r} (after filters).")
        return

    print(f"[info] loaded {len(runs)} runs from {args.file!r}")
    if args.feature is not None:
        print(f"[info] restricted to feature_id={args.feature}")
    if args.prompt_substr is not None:
        print(f"[info] restricted to prompts containing {args.prompt_substr!r}")

    print_overall_top(runs, k=args.top_overall)
    print_top_per_feature(runs, k=args.top_per_feature)
    print_example_snippets(runs, k=min(5, args.top_overall))


if __name__ == "__main__":
    main()