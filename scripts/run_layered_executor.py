#!/usr/bin/env python
r"""
run_layered_executor.py

Dry run:
    python run_layered_executor.py --config run_layered_config.json --dry-run

Run all jobs:
    python run_layered_executor.py --config run_layered_config.json

Run selected jobs:
    python run_layered_executor.py --config run_layered_config.json --only C57BL6J,AJ

Resume only unfinished jobs:
    python run_layered_executor.py --config run_layered_config.json --resume

Run up to 2 jobs at once:
    python run_layered_executor.py --config run_layered_config.json --parallel 2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Mapping, Optional


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON config without dependencies; load YAML when PyYAML is available."""
    suffix = config_path.suffix.lower()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install it with: pip install pyyaml\n"
                "Or use run_layered_config.json, which requires no extra package."
            ) from exc
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    raise ValueError(f"Unsupported config extension: {config_path.suffix}. Use .json, .yaml, or .yml")


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    return safe.strip("_") or "job"


def shell_join(cmd: Iterable[str]) -> str:
    """Readable command string for logs and dry-run output."""
    if os.name == "nt":
        return subprocess.list2cmdline(list(cmd))
    return " ".join(shlex.quote(part) for part in cmd)


def arg_to_cli(option_name: str, value: Any) -> List[str]:
    """
    Convert a config key/value pair into CLI args.

    Examples:
        {"model_file": "iMM1415.json"} -> ["--model_file", "iMM1415.json"]
        {"no_fva": true}               -> ["--no_fva"]
        {"aggregate": false}           -> []
        {"rank_product_for": ["HFD", "KD", "WD"]} -> ["--rank_product_for", "HFD,KD,WD"]
    """
    if value is None or value is False:
        return []

    flag = f"--{option_name}"

    if value is True:
        return [flag]

    if isinstance(value, (list, tuple)):
        # The original command expects rank_product_for as HFD,KD,WD.
        return [flag, ",".join(str(item) for item in value)]

    return [flag, str(value)]


def build_command(config: Mapping[str, Any], job: Mapping[str, Any]) -> List[str]:
    runner = config.get("runner", {})
    common_args = dict(config.get("common_args", {}))

    python_exe = str(runner.get("python_executable", "python"))
    script = str(runner.get("script", "map_fixv5_multigroupsv8_layered_manuscript_run.py"))

    input_csv = job.get("input_csv")
    if not input_csv:
        raise ValueError(f"Job {job.get('name', '<unnamed>')} is missing input_csv")

    merged_args: Dict[str, Any] = {}
    merged_args.update(common_args)

    # Allow a job to override any common parameter through a nested "args" block.
    merged_args.update(job.get("args", {}))

    # results_dir is required for this workflow and is usually job-specific.
    if "results_dir" in job:
        merged_args["results_dir"] = job["results_dir"]

    cmd = [python_exe, script, str(input_csv)]
    for key, value in merged_args.items():
        cmd.extend(arg_to_cli(key, value))
    return cmd


def project_path(project_root: Path, maybe_relative: Optional[str]) -> Optional[Path]:
    if maybe_relative is None:
        return None
    p = Path(maybe_relative)
    return p if p.is_absolute() else project_root / p


def success_marker_path(config: Mapping[str, Any], job: Mapping[str, Any], project_root: Path) -> Path:
    runner = config.get("runner", {})
    marker_name = runner.get("success_marker", ".runner_success.json")
    results_dir = project_path(project_root, str(job.get("results_dir", "")))
    if results_dir is None:
        raise ValueError(f"Job {job.get('name', '<unnamed>')} is missing results_dir")
    return results_dir / str(marker_name)


def preflight_job(config: Mapping[str, Any], job: Mapping[str, Any], project_root: Path) -> List[str]:
    """Return a list of warnings/errors discovered before running a job."""
    runner = config.get("runner", {})
    issues: List[str] = []

    script_path = project_path(project_root, str(runner.get("script", "")))
    input_path = project_path(project_root, str(job.get("input_csv", "")))

    common_args = config.get("common_args", {})

    files_to_check = [
        ("script", script_path),
        ("input_csv", input_path),
        ("model_file", project_path(project_root, common_args.get("model_file"))),
        ("diet_bounds_json", project_path(project_root, common_args.get("diet_bounds_json"))),
        ("mapping_file", project_path(project_root, common_args.get("mapping_file"))),
    ]

    for label, path in files_to_check:
        if path is not None and not path.exists():
            issues.append(f"{label} not found: {path}")

    return issues


def run_one_job(
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    project_root: Path,
    log_dir: Path,
    dry_run: bool = False,
    resume: bool = False,
    no_preflight: bool = False,
) -> Dict[str, Any]:
    job_name = str(job.get("name", "unnamed"))
    safe_name = sanitize_name(job_name)
    cmd = build_command(config, job)
    cmd_text = shell_join(cmd)

    result: Dict[str, Any] = {
        "job": job_name,
        "input_csv": job.get("input_csv", ""),
        "results_dir": job.get("results_dir", ""),
        "command": cmd_text,
        "status": "pending",
        "return_code": "",
        "started_at": "",
        "ended_at": "",
        "elapsed_seconds": "",
        "log_file": "",
    }

    marker = success_marker_path(config, job, project_root)
    if resume and marker.exists():
        result["status"] = "skipped_success_marker_exists"
        return result

    if dry_run:
        result["status"] = "dry_run"
        print(f"\n[{job_name}]")
        print(cmd_text)
        return result

    if not no_preflight:
        issues = preflight_job(config, job, project_root)
        if issues:
            result["status"] = "preflight_failed"
            result["return_code"] = "preflight"
            result["log_file"] = ""
            result["preflight_issues"] = "; ".join(issues)
            return result

    log_dir.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    # Ensure per-job output directory exists before running.
    if job.get("results_dir"):
        output_dir = project_path(project_root, str(job["results_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{now_stamp()}_{safe_name}.log"
    result["log_file"] = str(log_path)

    start = time.time()
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    result["started_at"] = started_at

    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write(f"Job: {job_name}\n")
        log.write(f"Started: {started_at}\n")
        log.write(f"Working directory: {project_root.resolve()}\n")
        log.write(f"Command: {cmd_text}\n")
        log.write("=" * 100 + "\n\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()

        ended_at = dt.datetime.now().isoformat(timespec="seconds")
        elapsed = round(time.time() - start, 2)

        log.write("\n" + "=" * 100 + "\n")
        log.write(f"Ended: {ended_at}\n")
        log.write(f"Elapsed seconds: {elapsed}\n")
        log.write(f"Return code: {return_code}\n")

    result["ended_at"] = ended_at
    result["elapsed_seconds"] = elapsed
    result["return_code"] = return_code
    result["status"] = "success" if return_code == 0 else "failed"

    if return_code == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "job": job_name,
                    "completed_at": ended_at,
                    "elapsed_seconds": elapsed,
                    "command": cmd_text,
                    "log_file": str(log_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return result


def filter_jobs(jobs: List[Mapping[str, Any]], only: Optional[str], exclude: Optional[str]) -> List[Mapping[str, Any]]:
    selected = jobs

    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        selected = [job for job in selected if str(job.get("name", "")) in wanted]

    if exclude:
        blocked = {item.strip() for item in exclude.split(",") if item.strip()}
        selected = [job for job in selected if str(job.get("name", "")) not in blocked]

    return selected


def write_summary(log_dir: Path, results: List[Mapping[str, Any]]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / f"run_summary_{now_stamp()}.csv"

    fields = [
        "job",
        "status",
        "return_code",
        "elapsed_seconds",
        "input_csv",
        "results_dir",
        "log_file",
        "started_at",
        "ended_at",
        "command",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return summary_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run layered manuscript model jobs from a config file.")
    parser.add_argument("--config", default="run_layered_config.json", help="Path to JSON/YAML config file.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--only", help="Comma-separated job names to run, e.g. C57BL6J,AJ")
    parser.add_argument("--exclude", help="Comma-separated job names to skip.")
    parser.add_argument("--parallel", type=int, help="Override config runner.parallel. Use carefully for memory-heavy runs.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with an existing success marker in their results_dir.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running after a failed job.")
    parser.add_argument("--no-preflight", action="store_true", help="Do not check for required files before running.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)

    runner = config.get("runner", {})
    project_root = Path(str(runner.get("project_root", "."))).resolve()
    log_dir = project_path(project_root, str(runner.get("log_dir", "logs")))
    if log_dir is None:
        raise ValueError("runner.log_dir cannot be empty")

    jobs = list(config.get("jobs", []))
    if not jobs:
        print("No jobs found in config.", file=sys.stderr)
        return 2

    jobs = filter_jobs(jobs, args.only, args.exclude)
    if not jobs:
        print("No jobs selected. Check --only/--exclude values.", file=sys.stderr)
        return 2

    parallel = args.parallel if args.parallel is not None else int(runner.get("parallel", 1))
    parallel = max(1, parallel)

    stop_on_error = bool(runner.get("stop_on_error", True)) and not args.continue_on_error

    print(f"Selected jobs: {', '.join(str(job.get('name', 'unnamed')) for job in jobs)}")
    print(f"Project root: {project_root}")
    print(f"Log directory: {log_dir}")
    print(f"Parallel jobs: {parallel}")
    if args.dry_run:
        print("Mode: dry run")

    results: List[Mapping[str, Any]] = []

    if args.dry_run or parallel == 1:
        for job in jobs:
            result = run_one_job(
                config=config,
                job=job,
                project_root=project_root,
                log_dir=log_dir,
                dry_run=args.dry_run,
                resume=args.resume,
                no_preflight=args.no_preflight,
            )
            results.append(result)
            print(f"{result['job']}: {result['status']}")

            if stop_on_error and result["status"] in {"failed", "preflight_failed"}:
                print("Stopping because a job failed. Use --continue-on-error to keep going.", file=sys.stderr)
                break
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            future_to_job = {
                pool.submit(
                    run_one_job,
                    config,
                    job,
                    project_root,
                    log_dir,
                    False,
                    args.resume,
                    args.no_preflight,
                ): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                print(f"{result['job']}: {result['status']}")

        # In parallel mode, all submitted jobs finish. We return non-zero if any failed.

    summary_path = write_summary(log_dir, results)
    print(f"\nSummary written to: {summary_path}")

    failed = [row for row in results if row["status"] in {"failed", "preflight_failed"}]
    if failed:
        print("\nFailed jobs:")
        for row in failed:
            print(f"  - {row['job']} ({row['status']}), log: {row.get('log_file', '')}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
