"""Overnight hyperparameter sweep driver for the repeater-chain surrogate.

Each trial shells out to ``train_surrogate_chain.py train`` with one set of overrides, then scores the
resulting model against every held-out testing condition. Results are appended to a CSV after each
trial, so an interrupted sweep keeps everything it already finished and resumes where it stopped.

Validation NLL alone is a misleading ranking signal here: the surrogate's weakest regime is short
chains, whose fidelity distribution is very sharp and contributes little to the pooled likelihood.
Every trial therefore also reports short-chain fidelity agreement, including the logit-space spread
ratio that measures over-dispersion directly.

The sweep is designed to run in two stages within one night. Stage one screens every configuration
at the shared base epoch count; stage two re-runs only the promising ones for much longer, using
``--epochs`` to override upward. The ``epoch_ratio`` column reports which screened configurations were
still improving when they stopped, which is the signal for what to promote.

Examples:
    python tune_surrogate_chain.py --list
    python tune_surrogate_chain.py --dry-run
    python tune_surrogate_chain.py                                  # stage one, screen everything
    python tune_surrogate_chain.py --only combo_wide_deep wide_128 components_24 --epochs 800 --force
    python tune_surrogate_chain.py --only base wide_128 --epoch-limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chain_metrics import score_conditions
from train_surrogate_chain import build_condition_sampler, load_model, resolve_device

# Shared starting point. Keys are train_surrogate_chain.py flag names without the leading dashes.
# Epochs are a screening length, chosen so every configuration below fits in one evening. Existing
# models all peak between epoch 163 and 192 of 200, so training was still improving when it stopped;
# the winners of this stage are meant to be re-run far longer with --epochs.
BASE_ARGUMENTS: dict[str, Any] = {
    "epochs": 250,
    "learning-rate": 3e-4,
    "dropout": 0.0,
    "components": 12,
    "embedding-dim": 64,
    "attention-heads": 4,
    "feedforward-dim": 128,
    "hidden-size": 64,
    "hidden-layers": 2,
    "transformer-layers": 2,
}

# Ordered by expected information per hour, so a truncated sweep still answers the important questions.
# Embedding dimensions stay divisible by their attention-head count.
TRIALS: list[tuple[str, dict[str, Any]]] = [
    ("base", {}),
    ("current_defaults", {"epochs": 200, "learning-rate": 1e-4, "dropout": 0.02}),
    ("learning_rate_1e-3", {"learning-rate": 1e-3}),
    ("wide_128", {"embedding-dim": 128, "attention-heads": 8, "feedforward-dim": 512, "hidden-size": 256}),
    ("batch_16", {"batch-size": 16}),
    ("components_24", {"components": 24}),
    ("schedule_threshold_1e-4", {"lr-threshold": 1e-4}),
    ("combo_wide_deep", {"embedding-dim": 128, "attention-heads": 8, "feedforward-dim": 512, "hidden-size": 256,
                         "transformer-layers": 3, "components": 24}),
    ("learning_rate_1e-4", {"learning-rate": 1e-4}),
    ("transformer_4", {"transformer-layers": 4}),
    ("batch_64", {"batch-size": 64}),
    ("huge", {"embedding-dim": 256, "attention-heads": 8, "feedforward-dim": 1024, "hidden-size": 512,
              "transformer-layers": 4, "components": 32}),
    ("hidden_3x256", {"hidden-size": 256, "hidden-layers": 3}),
    ("schedule_patience_25", {"lr-patience": 25}),
    ("components_40", {"components": 40}),
    ("gradient_clip_5", {"gradient-clip": 5.0}),
    ("learning_rate_5e-4", {"learning-rate": 5e-4}),
    ("wide_192", {"embedding-dim": 192, "attention-heads": 8, "feedforward-dim": 768, "hidden-size": 256}),
    ("combo_big_lr", {"embedding-dim": 128, "attention-heads": 8, "feedforward-dim": 512, "hidden-size": 256,
                      "learning-rate": 1e-3}),
    ("combo_big_lr_patience_25", {"embedding-dim": 128, "attention-heads": 8, "feedforward-dim": 512,
                                  "hidden-size": 256, "learning-rate": 1e-3, "lr-patience": 25}),
    ("huge_lr_1e-3", {"embedding-dim": 256, "attention-heads": 8, "feedforward-dim": 1024, "hidden-size": 512,
                      "transformer-layers": 4, "components": 32, "learning-rate": 1e-3}),
    ("dropout_002", {"dropout": 0.02}),
    ("weight_decay_1e-4", {"weight-decay": 1e-4}),
    ("heads_8", {"attention-heads": 8}),
    ("feedforward_512", {"feedforward-dim": 512}),
    ("combo_fast", {"learning-rate": 1e-3, "batch-size": 16}),
    ("components_8", {"components": 8}),
    ("transformer_1", {"transformer-layers": 1}),
    ("hidden_4x512", {"hidden-size": 512, "hidden-layers": 4}),
    ("combo_schedule", {"embedding-dim": 128, "attention-heads": 8, "feedforward-dim": 512, "hidden-size": 256,
                        "lr-patience": 25, "lr-threshold": 1e-4}),
    ("weight_decay_0", {"weight-decay": 0.0}),
]

# Per-epoch cost relative to the base configuration, for trials that differ enough to matter. Measured
# on an A30: training is step-bound at roughly 20 ms per optimizer step, so halving the batch size
# nearly doubles epoch time, and width costs far less than the parameter count suggests.
TRIAL_COST: dict[str, float] = {
    "batch_16": 2.0, "combo_fast": 2.0, "batch_64": 0.6,
    "wide_128": 1.3, "wide_192": 1.4, "combo_big_lr": 1.3, "combo_schedule": 1.3,
    "combo_big_lr_patience_25": 1.3, "huge_lr_1e-3": 1.6,
    "combo_wide_deep": 1.4, "huge": 1.6, "transformer_4": 1.2, "transformer_1": 0.85,
    "hidden_3x256": 1.1, "hidden_4x512": 1.2, "feedforward_512": 1.1,
    "components_24": 1.05, "components_40": 1.1, "components_8": 0.95,
}

# Measured on an A30: reading the 10,000 training files dominates startup, and per-epoch cost barely
# moves with model size because the GPU is nowhere near saturated. Refined at runtime from real trials.
STARTUP_SECONDS = 90.0
SECONDS_PER_EPOCH = 5.0
SHORT_CHAIN_MAX_NODES = 4
RESULT_FIELDS = [
    "name", "status", "timestamp", "train_seconds", "evaluate_seconds", "model_path",
    "epochs", "best_epoch", "epoch_ratio", "best_validation_nll", "best_train_nll",
    "final_learning_rate", "lr_reductions", "schedule_exhausted",
    "short_fidelity_ks", "short_logit_fidelity_std_ratio", "short_time_ks", "short_condition_count",
    "median_fidelity_ks", "p90_fidelity_ks", "median_logit_fidelity_std_ratio",
    "median_normalized_time_wasserstein", "p90_normalized_time_wasserstein", "median_time_ks",
    "median_joint_energy", "condition_count", "overrides", "note",
]
RANKING_COLUMNS = ["name", "status", "best_validation_nll", "short_fidelity_ks",
                   "short_logit_fidelity_std_ratio", "median_fidelity_ks",
                   "median_normalized_time_wasserstein", "epoch_ratio", "train_seconds"]


def trial_arguments(overrides: dict[str, Any], epochs: int | None, epoch_limit: int | None) -> dict[str, Any]:
    """Merge one trial's overrides onto the shared base arguments.

    Args:
        overrides (dict[str, Any]): Flag values specific to this trial.
        epochs (int | None): Sweep-wide epoch count replacing the base and per-trial values.
        epoch_limit (int | None): Optional cap applied last, for smoke tests.

    Returns:
        dict[str, Any]: Complete flag set for this trial.
    """
    arguments = {**BASE_ARGUMENTS, **overrides}
    if epochs is not None:
        arguments["epochs"] = epochs
    if epoch_limit is not None:
        arguments["epochs"] = min(int(arguments["epochs"]), epoch_limit)
    return arguments


def build_command(arguments: dict[str, Any], output: Path, inputs: list[Path], device: str, seed: int) -> list[str]:
    """Build the training subprocess command for one trial.

    Args:
        arguments (dict[str, Any]): Resolved training flags.
        output (Path): Requested model output path.
        inputs (list[Path]): Training data directories.
        device (str): Torch device string.
        seed (int): Training seed.

    Returns:
        list[str]: Argument vector for subprocess.run.
    """
    command = [sys.executable, "train_surrogate_chain.py", "train",
               "--input", *[str(path) for path in inputs],
               "--output", str(output), "--device", device, "--seed", str(seed)]
    for flag, value in arguments.items():
        command += [f"--{flag}", str(value)]
    return command


def parse_written_model(log_path: Path) -> Path | None:
    """Recover the saved model path from a training log.

    Training renames its output to embed the validation NLL, so the requested path is not the one on disk.

    Args:
        log_path (Path): Captured training log.

    Returns:
        Path | None: Saved model path, or None when the log has no completion line.
    """
    marker = "Wrote model:"
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith(marker):
            return Path(line[len(marker):].strip())
    return None


def read_training_metadata(model_path: Path) -> dict[str, Any]:
    """Read the JSON sidecar written next to a saved model.

    Args:
        model_path (Path): Saved model artifact path.

    Returns:
        dict[str, Any]: Parsed metadata, or an empty dict when the sidecar is missing.
    """
    sidecar = model_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-condition metrics into the columns used to rank trials.

    Args:
        records (list[dict[str, Any]]): Per-condition scores from chain_metrics.score_conditions.

    Returns:
        dict[str, float]: Aggregate metrics, overall and restricted to short chains.
    """
    def column(rows: list[dict[str, Any]], name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    short = [row for row in records if row["nodes"] <= SHORT_CHAIN_MAX_NODES]
    summary = {
        "condition_count": len(records),
        "short_condition_count": len(short),
        "median_fidelity_ks": float(np.nanmedian(column(records, "fidelity_ks"))),
        "p90_fidelity_ks": float(np.nanpercentile(column(records, "fidelity_ks"), 90)),
        "median_logit_fidelity_std_ratio": float(np.nanmedian(column(records, "logit_fidelity_std_ratio"))),
        "median_normalized_time_wasserstein": float(np.nanmedian(column(records, "normalized_time_wasserstein"))),
        "p90_normalized_time_wasserstein": float(np.nanpercentile(column(records, "normalized_time_wasserstein"), 90)),
        "median_time_ks": float(np.nanmedian(column(records, "time_ks"))),
        "median_joint_energy": float(np.nanmedian(column(records, "joint_energy"))),
    }
    if short:
        summary["short_fidelity_ks"] = float(np.nanmedian(column(short, "fidelity_ks")))
        summary["short_logit_fidelity_std_ratio"] = float(np.nanmedian(column(short, "logit_fidelity_std_ratio")))
        summary["short_time_ks"] = float(np.nanmedian(column(short, "time_ks")))
    else:
        summary.update({"short_fidelity_ks": float("nan"), "short_logit_fidelity_std_ratio": float("nan"),
                        "short_time_ks": float("nan")})
    return summary


def evaluate_model(model_path: Path, test_paths: list[Path], device: torch.device, sample_count: int,
                   seed: int) -> dict[str, float]:
    """Score one trained model against every held-out testing condition.

    Args:
        model_path (Path): Saved model artifact path.
        test_paths (list[Path]): Testing condition files.
        device (torch.device): Device used for sampling.
        sample_count (int): Surrogate samples drawn per condition.
        seed (int): Sampling seed.

    Returns:
        dict[str, float]: Aggregate metrics for this model.
    """
    artifact = load_model(model_path, device)
    sampler = build_condition_sampler(artifact, device)
    try:
        return summarize_records(score_conditions(test_paths, sampler, sample_count, seed))
    finally:
        del sampler, artifact
        if device.type == "cuda":
            torch.cuda.empty_cache()


def read_completed(results_path: Path) -> dict[str, dict[str, str]]:
    """Read trials already recorded by an earlier run of this sweep.

    Args:
        results_path (Path): Sweep results CSV.

    Returns:
        dict[str, dict[str, str]]: Recorded rows keyed by trial name.
    """
    if not results_path.exists():
        return {}
    with results_path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle) if row.get("name")}


def append_result(results_path: Path, row: dict[str, Any]):
    """Append one trial result, writing the header when the file is new.

    Args:
        results_path (Path): Sweep results CSV.
        row (dict[str, Any]): Result row to append.
    """
    header = None
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)

    if header is not None and header != RESULT_FIELDS:
        with results_path.open(newline="", encoding="utf-8") as handle:
            recorded = list(csv.DictReader(handle))
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({field: old.get(field, "") for field in RESULT_FIELDS} for old in recorded)
        header = RESULT_FIELDS

    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if header is None:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render selected columns of the result rows as a fixed-width table.

    Args:
        rows (list[dict[str, Any]]): Result rows.
        columns (list[str]): Column names to display, in order.

    Returns:
        str: Printable table.
    """
    def cell(value: Any) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
        return str(value)

    table = [columns] + [[cell(row.get(name, "")) for name in columns] for row in rows]
    widths = [max(len(line[index]) for line in table) for index in range(len(columns))]
    lines = ["  ".join(value.ljust(width) for value, width in zip(line, widths)).rstrip() for line in table]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def run_trial(name: str, overrides: dict[str, Any], test_paths: list[Path], device: torch.device,
              args: argparse.Namespace) -> dict[str, Any]:
    """Train one configuration and score the resulting model.

    Args:
        name (str): Trial name.
        overrides (dict[str, Any]): Flags overriding the base arguments.
        test_paths (list[Path]): Testing condition files.
        device (torch.device): Device used for evaluation sampling.
        args (argparse.Namespace): Sweep options.

    Returns:
        dict[str, Any]: Result row for this trial.
    """
    arguments = trial_arguments(overrides, args.epochs, args.epoch_limit)
    model_directory = args.output_directory / "models"
    log_path = args.output_directory / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    command = build_command(arguments, model_directory / f"{name}.pt", args.input, args.device, args.seed)
    row: dict[str, Any] = {"name": name, "timestamp": datetime.now().isoformat(timespec="seconds"),
                           "epochs": arguments["epochs"], "overrides": json.dumps(overrides, sort_keys=True)}

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    row["train_seconds"] = round(time.perf_counter() - started, 1)

    if completed.returncode != 0:
        return {**row, "status": "failed", "note": f"training exited with code {completed.returncode}; see {log_path}"}

    model_path = parse_written_model(log_path)
    if model_path is None or not model_path.exists():
        return {**row, "status": "failed", "note": f"no saved model found in {log_path}"}
    row["model_path"] = str(model_path)

    metadata = read_training_metadata(model_path)
    metrics = metadata.get("metrics", {})
    best_epoch = metrics.get("best_epoch")
    row["best_epoch"] = best_epoch
    row["best_validation_nll"] = metrics.get("best_validation_nll")
    row["best_train_nll"] = metrics.get("best_train_nll")
    if best_epoch:
        row["epoch_ratio"] = round(float(best_epoch) / float(arguments["epochs"]), 3)

    # A run that never reached the scheduler's floor still had annealing left, which is a far better
    # signal of under-training than the best epoch landing near the end: once the learning rate
    # bottoms out the model stops moving, and the best epoch after that is noise.
    learning_rates = metrics.get("learning_rate") or []
    if learning_rates:
        minimum = float(metadata.get("min_learning_rate", 1e-6))
        row["final_learning_rate"] = f"{learning_rates[-1]:.3e}"
        row["lr_reductions"] = sum(1 for before, after in zip(learning_rates, learning_rates[1:]) if after < before)
        row["schedule_exhausted"] = int(float(learning_rates[-1]) <= minimum * 1.001)

    started = time.perf_counter()
    try:
        row.update(evaluate_model(model_path, test_paths, device, args.sample_count, args.seed))
    except Exception as error:  # A single bad model must not end the overnight sweep.
        row["evaluate_seconds"] = round(time.perf_counter() - started, 1)
        return {**row, "status": "evaluation_failed", "note": str(error)}
    row["evaluate_seconds"] = round(time.perf_counter() - started, 1)
    return {**row, "status": "ok"}


def select_trials(only: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    """Restrict the trial list to explicitly requested names.

    Args:
        only (list[str] | None): Requested trial names, or None for the full list.

    Returns:
        list[tuple[str, dict[str, Any]]]: Trials to run, in sweep order.
    """
    if not only:
        return TRIALS
    known = {name for name, _ in TRIALS}
    unknown = [name for name in only if name not in known]
    if unknown:
        raise SystemExit(f"unknown trial name(s): {', '.join(unknown)}")
    return [(name, overrides) for name, overrides in TRIALS if name in set(only)]


def run_sweep(args: argparse.Namespace):
    """Run every selected trial within the time budget and print a ranked summary.

    Args:
        args (argparse.Namespace): Sweep options.
    """
    trials = select_trials(args.only)
    results_path = args.output_directory / "results.csv"
    completed = read_completed(results_path)

    if args.list or args.dry_run:
        planned = [(name, overrides, trial_arguments(overrides, args.epochs, args.epoch_limit)["epochs"])
                   for name, overrides in trials]
        rows = [{"name": name, "status": completed.get(name, {}).get("status", "pending"), "epochs": epochs,
                 "estimated_minutes": f"{estimate_seconds(epochs, TRIAL_COST.get(name, 1.0), 1.0) / 60:.0f}",
                 "overrides": json.dumps(overrides, sort_keys=True)} for name, overrides, epochs in planned]
        total = sum(float(row["estimated_minutes"]) for row in rows if row["status"] != "ok")
        print(format_table(rows, ["name", "status", "epochs", "estimated_minutes", "overrides"]))
        print(f"\nPending trials estimated at {total / 60:.1f} h against a {args.time_budget_hours:.1f} h budget.")
        if args.dry_run:
            print(f"Dry run: nothing trained. Results would be appended to {results_path}.")
        return

    test_paths = sorted(args.test_directory.glob("*.h5"))
    if not test_paths:
        raise SystemExit(f"No HDF5 files found under {args.test_directory}")
    device = resolve_device(args.device)
    deadline = time.time() + args.time_budget_hours * 3600
    calibration = 1.0
    observed: list[tuple[float, float]] = []

    print(f"Sweep of {len(trials)} trial(s) on {len(test_paths)} testing conditions, device {device}")
    print(f"Budget {args.time_budget_hours:.1f} h, results appended to {results_path}\n")

    for index, (name, overrides) in enumerate(trials, start=1):
        if completed.get(name, {}).get("status") == "ok" and not args.force:
            print(f"[{index}/{len(trials)}] {name}: already recorded, skipping")
            continue

        epochs = trial_arguments(overrides, args.epochs, args.epoch_limit)["epochs"]
        predicted = estimate_seconds(epochs, TRIAL_COST.get(name, 1.0), calibration)
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"[{index}/{len(trials)}] {name}: budget exhausted, stopping")
            break
        if predicted > remaining:
            print(f"[{index}/{len(trials)}] {name}: needs ~{predicted / 60:.0f} min but only "
                  f"{remaining / 60:.0f} min left, skipping")
            continue

        print(f"[{index}/{len(trials)}] {name}: {epochs} epochs, ~{predicted / 60:.0f} min predicted, "
              f"{remaining / 3600:.1f} h left")
        row = run_trial(name, overrides, test_paths, device, args)
        append_result(results_path, row)
        completed[name] = {key: str(value) for key, value in row.items()}

        note = f" ({row['note']})" if row.get("note") else ""
        print(f"    {row['status']} in {row['train_seconds'] / 60:.1f} min{note}")
        if row["status"] == "ok":
            print(f"    validation NLL {row['best_validation_nll']:.4f}, best epoch {row['best_epoch']}"
                  f"/{epochs}, short-chain fidelity KS {row['short_fidelity_ks']:.4f}, "
                  f"logit sigma ratio {row['short_logit_fidelity_std_ratio']:.3f}")
        if row["status"] != "failed":  # A crashed run says nothing about how long a real one takes.
            observed.append((estimate_seconds(epochs, TRIAL_COST.get(name, 1.0), 1.0), float(row["train_seconds"])))
            calibration = float(np.mean([actual / base for base, actual in observed]))

    print_summary(results_path)


def print_summary(results_path: Path):
    """Print every recorded trial ranked by short-chain fidelity agreement.

    Args:
        results_path (Path): Sweep results CSV.
    """
    rows = list(read_completed(results_path).values())
    if not rows:
        print("\nNo results recorded.")
        return

    def numeric(row: dict[str, str], name: str) -> float:
        try:
            return float(row.get(name, "") or "nan")
        except ValueError:
            return float("nan")

    parsed = [{**row, **{name: numeric(row, name) for name in RANKING_COLUMNS if name not in ("name", "status")}}
              for row in rows]
    ok = sorted((row for row in parsed if row["status"] == "ok"),
                key=lambda row: (np.isnan(row["short_fidelity_ks"]), row["short_fidelity_ks"]))
    other = [row for row in parsed if row["status"] != "ok"]

    print(f"\n{'=' * 100}\nTrials ranked by short-chain (<= {SHORT_CHAIN_MAX_NODES} nodes) fidelity KS, "
          f"lower is better\n{'=' * 100}")
    print(format_table(ok + other, RANKING_COLUMNS))

    unfinished = [row["name"] for row in ok if str(row.get("schedule_exhausted", "")) == "0"]
    if unfinished:
        print(f"\nStopped before the learning rate reached its floor: {', '.join(unfinished)}")
        print("Those schedules still had annealing left, so their scores are a lower bound. Re-run them with "
              "--epochs. Trials not listed here converged; a best epoch near the end is noise at a dead "
              "learning rate, not evidence of under-training.")
    print(f"\nFull per-trial results: {results_path}")


def estimate_seconds(epochs: int, cost: float, calibration: float) -> float:
    """Estimate wall-clock seconds for one training run.

    Args:
        epochs (int): Epoch count for the trial.
        cost (float): Per-epoch cost of this trial relative to the base configuration.
        calibration (float): Multiplier learned from trials already observed in this sweep.

    Returns:
        float: Estimated training seconds.
    """
    return (STARTUP_SECONDS + epochs * SECONDS_PER_EPOCH * cost) * calibration


def parse_arguments() -> argparse.Namespace:
    """Parse sweep options.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("chain/training/h5")],
                        help="Training data directories passed through to train_surrogate_chain.py.")
    parser.add_argument("--test-directory", type=Path, default=Path("chain/testing/h5"),
                        help="Held-out conditions used to score every trained model.")
    parser.add_argument("--output-directory", type=Path, default=Path("tuning"),
                        help="Destination for sweep models, logs, and results.csv. Kept separate from models/ so "
                             "the evaluation notebook never picks up a sweep model.")
    parser.add_argument("--time-budget-hours", type=float, default=14.0,
                        help="Stop starting new trials once the remaining time cannot fit one.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Replace every trial's epoch count, in either direction. Use with --only and --force "
                             "to re-run the best screened configurations for much longer.")
    parser.add_argument("--epoch-limit", type=int, default=None,
                        help="Cap every trial's epochs, applied after --epochs. Use for a quick smoke test.")
    parser.add_argument("--only", nargs="+", help="Run only these trial names.")
    parser.add_argument("--force", action="store_true", help="Rerun trials already recorded as ok.")
    parser.add_argument("--list", action="store_true", help="List the trials and their estimated cost, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Same as --list, and state that nothing will train.")
    parser.add_argument("--summary", action="store_true", help="Print the ranked summary of recorded results and exit.")
    parser.add_argument("--sample-count", type=int, default=2000, help="Surrogate samples drawn per test condition.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    """Run the hyperparameter sweep from the command line."""
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.summary:
        print_summary(args.output_directory / "results.csv")
        return
    run_sweep(args)


if __name__ == "__main__":
    main()
