#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tiny_decision_stack.backends import LocalLFMBackend, LocalOpenDeciderBackend
from tiny_decision_stack.credible_benchmark import (
    BOOTSTRAP_SEED,
    DEFAULT_THRESHOLDS,
    load_cases,
    paired_bootstrap_difference,
    select_threshold,
    sha256_file,
    simulate,
    summarize_by_task,
    summarize_outcomes,
    trace_case_pair,
    verify_suite,
)
from tiny_decision_stack.orchestrator import DecisionOrchestrator


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_hf_revision(model_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).sha
    except Exception:
        return None


def git_sha() -> str | None:
    value = os.getenv("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def collect_traces(orchestrator, cases, thresholds, predictions_handle, split_name: str):
    direct = []
    full = []
    max_threshold = max(thresholds)
    for idx, case in enumerate(cases, 1):
        d, f = trace_case_pair(orchestrator, case, max_threshold=max_threshold)
        direct.append(d)
        full.append(f)
        predictions_handle.write(json.dumps({"split": split_name, "trace": d.to_dict()}, ensure_ascii=False) + "\n")
        predictions_handle.write(json.dumps({"split": split_name, "trace": f.to_dict()}, ensure_ascii=False) + "\n")
        predictions_handle.flush()
        if idx == 1 or idx % 25 == 0 or idx == len(cases):
            print(f"[{split_name}] traced {idx}/{len(cases)} cases", flush=True)
    return direct, full


def pct(value):
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def ms(value):
    return f"{value:.1f} ms"


def render_markdown(report: dict) -> str:
    lines = [
        "# Credible model-accuracy benchmark v1",
        "",
        f"- Holdout cases: **{report['dataset']['holdout_count']}** (never used for threshold selection)",
        f"- Calibration cases: **{report['dataset']['calibration_count']}**",
        f"- Target selective precision: **{pct(report['config']['target_precision'])}**",
        f"- Bootstrap seed: `{report['comparison']['paired_bootstrap']['seed']}`",
        "",
        "## Headline holdout results",
        "",
        "| Variant | Calibrated threshold | Top-1 accuracy @ 0 | Effective accuracy | Selective precision | Coverage | Model errors | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("full_stack", "direct_only"):
        item = report["variants"][key]
        h = item["holdout"]
        raw = item["holdout_at_zero"]
        lines.append(
            f"| {key} | {item['calibration']['chosen_threshold']:.3f} | {pct(raw['effective_accuracy'])} | "
            f"{pct(h['effective_accuracy'])} | {pct(h['selective_precision'])} | {pct(h['coverage'])} | "
            f"{pct(h['model_error_rate'])} | {ms(h['p50_latency_ms'])} | {ms(h['p95_latency_ms'])} |"
        )
    boot = report["comparison"]["paired_bootstrap"]
    raw_boot = report["comparison"]["raw_top1_paired_bootstrap"]
    lo, hi = boot["ci95"]
    raw_lo, raw_hi = raw_boot["ci95"]
    lines += [
        "",
        "## Paired comparison",
        "",
        f"Calibrated selective-policy effective accuracy difference (full - direct): **{boot['observed_difference']:+.4f}** "
        f"(paired bootstrap 95% CI **[{lo:+.4f}, {hi:+.4f}]**, {boot['bootstrap_reps']} resamples).",
        f"Raw top-1 accuracy difference at threshold 0 (full - direct): **{raw_boot['observed_difference']:+.4f}** "
        f"(paired bootstrap 95% CI **[{raw_lo:+.4f}, {raw_hi:+.4f}]**).",
        "",
        "## Per-task holdout metrics at calibrated thresholds",
        "",
        "| Variant | Task | N | Effective accuracy | Selective precision | Coverage | Model errors | Clarification | Initial normalization |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("full_stack", "direct_only"):
        for task, m in report["variants"][key]["holdout_by_task"].items():
            lines.append(
                f"| {key} | {task} | {m['n']} | {pct(m['effective_accuracy'])} | {pct(m['selective_precision'])} | "
                f"{pct(m['coverage'])} | {pct(m['model_error_rate'])} | {pct(m['clarification_rate'])} | {pct(m['initial_normalization_rate'])} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Thresholds are selected **only on the calibration split**. The frozen holdout is evaluated exactly once at each preselected variant threshold. "
        "These public datasets may have appeared in model pretraining, so this is credible zero-shot benchmark evidence, not proof of contamination-free general intelligence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks/credible-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/credible-v1"))
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--limit-calibration", type=int, default=None, help="debug only; disables scientific comparability")
    parser.add_argument("--limit-holdout", type=int, default=None, help="debug only; disables scientific comparability")
    args = parser.parse_args()

    thresholds = tuple(float(v) for v in args.thresholds.split(","))
    if not 0.0 <= args.target_precision <= 1.0:
        raise ValueError("target precision must be in [0,1]")

    manifest = verify_suite(args.benchmark_dir)
    calibration = load_cases(args.benchmark_dir / "calibration.jsonl")
    holdout = load_cases(args.benchmark_dir / "holdout.jsonl")
    if args.limit_calibration:
        calibration = calibration[: args.limit_calibration]
    if args.limit_holdout:
        holdout = holdout[: args.limit_holdout]

    model_device = os.getenv("MODEL_DEVICE") or None
    lfm_id = os.getenv("LFM_MODEL", "LiquidAI/LFM2.5-350M")
    decider_id = os.getenv("OPENDECIDER_MODEL", "manjunathshiva/opendecider-nano")
    lfm_revision = resolve_hf_revision(lfm_id)
    decider_revision = resolve_hf_revision(decider_id)
    orchestrator = DecisionOrchestrator(
        semantic=LocalLFMBackend(lfm_id, model_device),
        decider=LocalOpenDeciderBackend(decider_id, model_device),
        direct_max_chars=int(os.getenv("DIRECT_MAX_CHARS", "500")),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "per_case_traces.jsonl"
    with predictions_path.open("w", encoding="utf-8") as pred:
        cal_direct, cal_full = collect_traces(orchestrator, calibration, thresholds, pred, "calibration")
        full_selection = select_threshold(cal_full, thresholds=thresholds, target_precision=args.target_precision, role="calibration")
        direct_selection = select_threshold(cal_direct, thresholds=thresholds, target_precision=args.target_precision, role="calibration")
        print("full_stack chosen threshold", full_selection["chosen_threshold"], "target_met", full_selection["target_met"], flush=True)
        print("direct_only chosen threshold", direct_selection["chosen_threshold"], "target_met", direct_selection["target_met"], flush=True)
        hold_direct, hold_full = collect_traces(orchestrator, holdout, thresholds, pred, "holdout")

    full_threshold = full_selection["chosen_threshold"]
    direct_threshold = direct_selection["chosen_threshold"]
    full_out = [simulate(trace, full_threshold) for trace in hold_full]
    direct_out = [simulate(trace, direct_threshold) for trace in hold_direct]
    full_zero = [simulate(trace, 0.0) for trace in hold_full]
    direct_zero = [simulate(trace, 0.0) for trace in hold_direct]

    report = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": git_sha(),
        "config": {
            "target_precision": args.target_precision,
            "thresholds": thresholds,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "debug_limits": {"calibration": args.limit_calibration, "holdout": args.limit_holdout},
        },
        "dataset": {
            "manifest_sha256": sha256_file(args.benchmark_dir / "manifest.json"),
            "calibration_sha256": sha256_file(args.benchmark_dir / "calibration.jsonl"),
            "holdout_sha256": sha256_file(args.benchmark_dir / "holdout.jsonl"),
            "calibration_count": len(calibration),
            "holdout_count": len(holdout),
            "upstream": manifest["sources"],
        },
        "models": {
            "semantic": {"id": lfm_id, "resolved_revision": lfm_revision},
            "decider": {"id": decider_id, "resolved_revision": decider_revision},
            "device": model_device or "auto",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ("tiny-decision-stack", "torch", "transformers", "accelerate", "opendecider")
            },
        },
        "variants": {
            "full_stack": {
                "calibration": full_selection,
                "holdout_at_zero": summarize_outcomes(full_zero),
                "holdout": summarize_outcomes(full_out),
                "holdout_by_task": summarize_by_task(full_out),
            },
            "direct_only": {
                "calibration": direct_selection,
                "holdout_at_zero": summarize_outcomes(direct_zero),
                "holdout": summarize_outcomes(direct_out),
                "holdout_by_task": summarize_by_task(direct_out),
            },
        },
        "comparison": {
            "paired_bootstrap": paired_bootstrap_difference(
                full_out, direct_out, reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED
            ),
            "raw_top1_paired_bootstrap": paired_bootstrap_difference(
                full_zero, direct_zero, reps=args.bootstrap_reps, seed=BOOTSTRAP_SEED
            ),
        },
        "limitations": manifest["limitations"],
    }

    report_path = args.output_dir / "report.json"
    md_path = args.output_dir / "summary.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {report_path}")
    print(f"wrote {predictions_path}")


if __name__ == "__main__":
    main()
