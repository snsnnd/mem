"""B1.3 protocol pilot: common replay, prequential scoring, raw traces, paired CIs.

This is NOT the completed learnable-controller experiment or an on-policy evaluation.
Run: python -m personal_cortex_lab.run_b1_3 --output results/b1_3_pilot
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

from .replay import Episode, SCENARIOS, VALUES, make_episode
from .replay_baselines import FACTORIES


def labels(distributions):
    # Fixed tie convention, identical for every method; latent values are randomized.
    return {d: VALUES[d][max(range(len(p)), key=p.__getitem__)]
            for d, p in distributions.items()}


def expected_answer_error(prediction, target, instruction=(), adherence=0.85):
    """Analytic version of B1.2's independent, per-dimension mock execution noise."""
    if not 0 <= adherence <= 1:
        raise ValueError("Adherence must be in [0, 1]")
    effective = {**prediction, **dict(instruction)}
    success = math.prod(adherence if effective[d] == target[d]
                        else (1 - adherence) / (len(v) - 1) for d, v in VALUES.items())
    return 1 - success


def evaluate(episode, learner, prediction_sink=None, method="", recovery_window=3):
    if recovery_window < 1:
        raise ValueError("Recovery window must be positive")
    totals = dict(correct=0, joint_errors=0, nll=0., brier=0., answer_error=0.,
                  changes=0, harmful=0)
    active = {}
    recoveries = []
    for step in episode.steps:
        obs = step.observation
        truth, target = dict(step.latent), dict(step.target)
        # Only task/time enter prediction; current feedback is unavailable until AFTER scoring.
        before = learner.predict(obs.task, obs.turn)
        pred = labels(before)
        correct = sum(pred[d] == truth[d] for d in VALUES)
        totals["correct"] += correct
        totals["joint_errors"] += correct != len(VALUES)
        totals["answer_error"] += expected_answer_error(pred, target, obs.instruction)
        for d, options in VALUES.items():
            p = before[d]
            if len(p) != len(options) or any(not math.isfinite(x) or x < 0 for x in p) or not math.isclose(sum(p), 1):
                raise ValueError(f"Invalid probability vector for {d}: {p}")
            totals["nll"] -= math.log(max(p[options.index(truth[d])], 1e-12))
            totals["brier"] += sum((x - (v == truth[d])) ** 2 for x, v in zip(p, options))
        for task, dim, value in step.changes:
            record = dict(task=task, dim=dim, value=value, start_turn=obs.turn,
                          visits=0, streak=0, recovered_turn=None, recovery_visits=None,
                          end_turn=len(episode.steps))
            key = task, dim
            if key in active:
                active[key]["end_turn"] = obs.turn
            active[key] = record
            recoveries.append(record)
        for (task, dim), record in active.items():
            if task == obs.task and record["recovered_turn"] is None:
                record["visits"] += 1
                record["streak"] = record["streak"] + 1 if pred[dim] == record["value"] else 0
                if record["streak"] >= recovery_window:
                    record["recovered_turn"] = obs.turn
                    record["recovery_visits"] = record["visits"]
        learner.observe(obs)  # Immutable, public Observation only: no latent, target or changes.
        after = learner.predict(obs.task, obs.turn)
        post = labels(after)
        for d in VALUES:
            changed = pred[d] != post[d]
            totals["changes"] += changed
            totals["harmful"] += changed and pred[d] == truth[d] and post[d] != truth[d]
        if prediction_sink is not None:
            prediction_sink.write(json.dumps(dict(episode=episode.key, method=method,
                                                   turn=obs.turn, before=before, after=after)) + "\n")
    turns = len(episode.steps)
    if not turns:
        raise ValueError("Cannot evaluate an empty episode")
    return {
        "hidden_acc": totals["correct"] / (turns * len(VALUES)),
        "joint_state_error": totals["joint_errors"] / turns,
        "expected_mock_answer_error": totals["answer_error"] / turns,
        "nll": totals["nll"] / (turns * len(VALUES)),
        "brier": totals["brier"] / (turns * len(VALUES)),
        "harmful_write_rate": totals["harmful"] / totals["changes"] if totals["changes"] else 0.,
        "harmful_writes": totals["harmful"],
        "prediction_changes": totals["changes"],
        "recovery_rate": (sum(r["recovered_turn"] is not None for r in recoveries) / len(recoveries)
                          if recoveries else None),
        "recoveries": recoveries,
    }


METRICS = ("hidden_acc", "joint_state_error", "expected_mock_answer_error", "nll", "brier",
           "harmful_write_rate", "recovery_rate")


def interval(values):
    """Percentile bootstrap of seed clusters, NOT independent time steps."""
    if not values:
        return None
    rng = random.Random(1301)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return dict(mean=statistics.mean(values),
                seed_sd=statistics.stdev(values) if len(values) > 1 else None,
                ci95=[samples[49], samples[1949]] if len(values) > 1 else None,
                n_seeds=len(values))


def summarize(rows):
    summary = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        group = [r for r in rows if r["scenario"] == scenario]
        methods = sorted({r["method"] for r in group})
        by_method = {}
        for method in methods:
            seed_means = {}
            for metric in METRICS:
                seed_means[metric] = {}
                for seed in sorted({r["seed"] for r in group}):
                    values = [r["metrics"][metric] for r in group if r["method"] == method and r["seed"] == seed
                              and r["metrics"][metric] is not None]
                    if values:
                        seed_means[metric][seed] = statistics.mean(values)
            by_method[method] = seed_means
        summary[scenario] = {}
        for method in methods:
            result = {}
            for metric in METRICS:
                values = by_method[method][metric]
                ref = by_method["Context-Latest"][metric]
                paired = [value - ref[seed] for seed, value in values.items() if seed in ref]
                result[metric] = dict(estimate=interval(list(values.values())),
                                      delta_vs_context_latest=interval(paired))
            summary[scenario][method] = result
    return summary


def episodes_from(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield Episode.from_dict(json.loads(line))


def provenance():
    package = Path(__file__).parent
    hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
              for name in ("replay.py", "replay_baselines.py", "run_b1_3.py")}
    root = package.parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return dict(python=platform.python_version(), git_revision=revision.stdout.strip(),
                git_dirty=bool(status.stdout.strip()), source_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--scenarios", nargs="+", choices=[s.name for s in SCENARIOS])
    parser.add_argument("--replay", type=Path, help="Replay saved episodes; generation flags are ignored")
    args = parser.parse_args()
    if args.users < 1 or args.turns < 20 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require users >= 1, turns >= 20, unique seeds")
    metadata = provenance()
    args.output.mkdir(parents=True, exist_ok=False)  # Never overwrite existing research artifacts.
    scenarios = [s for s in SCENARIOS if args.scenarios is None or s.name in args.scenarios]
    episodes = (episodes_from(args.replay) if args.replay else
                (make_episode(s, seed, user, args.turns, args.split)
                 for s in scenarios for seed in args.seeds for user in range(args.users)))
    rows = []
    trace_digest = hashlib.sha256()
    with gzip.open(args.output / "episodes.jsonl.gz", "wt", encoding="utf-8") as traces, \
            gzip.open(args.output / "predictions.jsonl.gz", "wt", encoding="utf-8") as predictions, \
            (args.output / "per_user.jsonl").open("w", encoding="utf-8") as metrics:
        for episode in episodes:
            serialized = json.dumps(episode.to_dict(), sort_keys=True) + "\n"
            traces.write(serialized)
            trace_digest.update(serialized.encode())
            for name, factory in FACTORIES.items():
                result = evaluate(episode, factory(), predictions, name)
                row = dict(episode=episode.key, scenario=episode.scenario.name, seed=episode.seed,
                           user=episode.user, split=episode.split, method=name, metrics=result)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
    metadata.update(schema="b1.3-replay-v1", protocol="exogenous-structured-reports",
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    scenarios=[asdict(s) for s in scenarios] if not args.replay else None,
                    trace_content_sha256=trace_digest.hexdigest(),
                    baselines={"Context-DiscountedCounts": {"discount": 0.98, "prior": 0.5},
                               "Context-Delta": {"eta": 0.4, "decay": 0.99},
                               "Context-HazardFilter": {"hazard": 0.01, "noise": 0.1}},
                    recovery_window=3, mock_adherence=0.85)
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for scenario, methods in summary.items():
        print(f"\n{scenario}")
        for name, result in methods.items():
            acc = result["hidden_acc"]["estimate"]["mean"]
            delta = result["hidden_acc"]["delta_vs_context_latest"]
            print(f"  {name:<25} hidden_acc={acc:.4f} delta={delta['mean']:+.4f} CI={delta['ci95']}")


if __name__ == "__main__":
    main()
