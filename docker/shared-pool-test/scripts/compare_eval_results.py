#!/usr/bin/env python3
"""
Aggregate the per-cell JSONLs produced by `run_eval.sh` into:
  * summary.csv  (long form: one row per JSONL)
  * summary.md   (pivoted: each (model, tp, mfs, workload) cell shows the
                  three variants side-by-side with deltas)

Filename convention emitted by run_eval.sh:

    <model_slug>__<variant>__tp<N>__mfs<MFS>__<workload>.jsonl

  * model_slug  : tiiuae_Falcon-H1-7B-Instruct (slashes -> underscores)
  * variant     : baseline_triton | baseline_native | shared
  * N           : 1 | 2 | ...
  * MFS         : e.g. 0.85 | 0.55
  * workload    : balanced | balanced-large | skewed | skewed-large

Usage:
    python3 compare_eval_results.py /path/to/eval_<model>_<ts>/

Designed to be standalone-runnable on the host or inside the container —
no third-party deps beyond stdlib.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Metrics extracted from each JSONL (keys are present in the bench_serving
# output across recent SGLang versions; missing keys produce blank cells).
METRICS_OF_INTEREST = [
    "request_throughput",
    "input_token_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "successful_requests",
    "duration",
]

VARIANTS_ORDER = ["baseline_triton", "baseline_native", "shared"]

FILENAME_RE = re.compile(
    r"^(?P<model>.+?)__(?P<variant>baseline_triton|baseline_native|shared)"
    r"__tp(?P<tp>\d+)"
    r"__mfs(?P<mfs>[0-9.]+)"
    r"__(?P<workload>[A-Za-z0-9-]+)\.jsonl$"
)


def parse_filename(p: Path) -> Optional[Dict[str, str]]:
    m = FILENAME_RE.match(p.name)
    if not m:
        return None
    return m.groupdict()


def load_last_record(p: Path) -> Optional[Dict]:
    """Read the LAST non-empty line of the JSONL (bench_serving may write
    multiple records; the last one is the final summary)."""
    try:
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def collect_rows(runs_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for jsonl_path in sorted(runs_dir.glob("*.jsonl")):
        meta = parse_filename(jsonl_path)
        if not meta:
            continue
        rec = load_last_record(jsonl_path)
        if rec is None:
            continue
        row = dict(meta)
        for k in METRICS_OF_INTEREST:
            row[k] = rec.get(k, "")
        row["_jsonl_path"] = str(jsonl_path)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("(no rows)\n")
        return
    keys = ["model", "variant", "tp", "mfs", "workload"] + METRICS_OF_INTEREST
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(v) -> str:
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) < 1:
            return f"{v:.4f}"
        if abs(v) < 100:
            return f"{v:.2f}"
        return f"{v:.1f}"
    return str(v)


def fmt_pct_delta(base, other) -> str:
    """Return '+5.6%' / '-3.2%' / '-' for a metric pair, or '-' on missing."""
    try:
        b = float(base)
        o = float(other)
    except (TypeError, ValueError):
        return "-"
    if b == 0:
        return "-"
    pct = (o - b) / b * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def write_markdown_summary(rows: List[Dict], out_path: Path) -> None:
    """Pivot rows into per-cell tables, one section per (model, tp, mfs, workload)."""
    # Bucket by (model, tp, mfs, workload).
    cells: Dict[Tuple[str, str, str, str], Dict[str, Dict]] = {}
    for r in rows:
        key = (r["model"], r["tp"], r["mfs"], r["workload"])
        cells.setdefault(key, {})[r["variant"]] = r

    if not cells:
        out_path.write_text("# Eval summary\n\n(no rows)\n")
        return

    out_lines: List[str] = []
    out_lines.append("# Eval summary\n")
    out_lines.append(
        "Variants: `baseline_triton` (static partition + Triton kernel) | "
        "`baseline_native` (static + native) | `shared` (shared pool, "
        "forced native).\n\n"
        "Pairwise deltas:\n"
        "* `triton → native` isolates the GPU-kernel-loss cost.\n"
        "* `native → shared` isolates the shared-pool bookkeeping cost.\n"
        "* `triton → shared` is the total shared-pool cost.\n\n"
    )

    for key in sorted(cells.keys()):
        model, tp, mfs, workload = key
        per_variant = cells[key]
        # Skip cells where we have no data at all.
        if not per_variant:
            continue

        out_lines.append(f"## {model} | tp={tp} | mfs={mfs} | workload={workload}\n\n")

        # Header:  metric | triton | native | shared | t→n | n→s | t→s
        header = (
            "| metric | baseline_triton | baseline_native | shared | "
            "Δ triton→native | Δ native→shared | Δ triton→shared |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n"
        )
        out_lines.append(header)

        bt = per_variant.get("baseline_triton")
        bn = per_variant.get("baseline_native")
        sh = per_variant.get("shared")

        for metric in METRICS_OF_INTEREST:
            v_t = bt[metric] if bt else ""
            v_n = bn[metric] if bn else ""
            v_s = sh[metric] if sh else ""
            d_tn = fmt_pct_delta(v_t, v_n) if (bt and bn) else "-"
            d_ns = fmt_pct_delta(v_n, v_s) if (bn and sh) else "-"
            d_ts = fmt_pct_delta(v_t, v_s) if (bt and sh) else "-"
            out_lines.append(
                f"| {metric} | {fmt_value(v_t)} | {fmt_value(v_n)} | {fmt_value(v_s)} | "
                f"{d_tn} | {d_ns} | {d_ts} |\n"
            )
        out_lines.append("\n")

    out_path.write_text("".join(out_lines))


def print_console_table(rows: List[Dict]) -> None:
    """Compact one-line-per-cell view to stdout — useful when summary.md
    is too long to scan and you just want headline numbers."""
    if not rows:
        print("(no rows)")
        return

    # Group by (model, tp, mfs, workload), one line per group with three
    # variants' total_token_throughput + median_itl_ms.
    cells: Dict[Tuple[str, str, str, str], Dict[str, Dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["tp"], r["mfs"], r["workload"]), {})[
            r["variant"]
        ] = r

    print()
    print(f"{'cell':<70} | {'metric':<22} | {'triton':>10} | {'native':>10} | {'shared':>10} | {'t→s%':>8}")
    print("-" * 150)
    for key in sorted(cells.keys()):
        model, tp, mfs, workload = key
        per_variant = cells[key]
        cell_label = f"{model} tp={tp} mfs={mfs} {workload}"
        for metric in ("total_token_throughput", "median_itl_ms"):
            v_t = per_variant.get("baseline_triton", {}).get(metric, "")
            v_n = per_variant.get("baseline_native", {}).get(metric, "")
            v_s = per_variant.get("shared", {}).get(metric, "")
            d_ts = fmt_pct_delta(v_t, v_s) if (v_t and v_s) else "-"
            print(
                f"{cell_label[:70]:<70} | {metric:<22} | "
                f"{fmt_value(v_t):>10} | {fmt_value(v_n):>10} | "
                f"{fmt_value(v_s):>10} | {d_ts:>8}"
            )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    eval_dir = Path(sys.argv[1])
    if not eval_dir.is_dir():
        print(f"ERROR: not a directory: {eval_dir}", file=sys.stderr)
        return 1

    runs_dir = eval_dir / "runs"
    if not runs_dir.is_dir():
        print(f"ERROR: no runs/ subdir at {runs_dir}", file=sys.stderr)
        return 1

    rows = collect_rows(runs_dir)
    print(f"Collected {len(rows)} run records from {runs_dir}")

    write_csv(rows, eval_dir / "summary.csv")
    write_markdown_summary(rows, eval_dir / "summary.md")
    print(f"Wrote {eval_dir / 'summary.csv'}")
    print(f"Wrote {eval_dir / 'summary.md'}")
    print_console_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
