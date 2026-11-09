"""B1.3 learned-controller experiment. Fit/selection and held-out test are separate commands.

python -m personal_cortex_lab.train_memory_controller fit --output results/b1_3_fit
python -m personal_cortex_lab.train_memory_controller test --fit results/b1_3_fit --output results/b1_3_test
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .learned_memory import (
    GateController, ScoringReplay, encode_observations, encode_targets, rollout,
)
from .replay import Episode, Evidence, Observation, SCENARIOS, TASKS, VALUES, Scenario, Step, make_episode
from .replay_baselines import DeltaMemory, DiscountedCounts, HazardFilter, Latest
from .run_b1_3 import METRICS, evaluate, interval, labels, provenance, summarize

TRAIN_SEEDS = tuple(range(1000, 1020))
VALIDATION_SEEDS = tuple(range(2000, 2010))
TEST_SEEDS = tuple(range(3000, 3020))
INITIALIZATIONS = (17, 29, 43)
OOD_SCENARIOS = (
    Scenario("ood_noise35", feedback_prob=0.3, noise_prob=0.35, drift="abrupt"),
    Scenario("ood_sparse10", feedback_prob=0.1, noise_prob=0.2, drift="abrupt"),
    Scenario("ood_renewal", feedback_prob=0.4, noise_prob=0.1),
)


def renewal_episode(scenario, seed, user, turns, split):
    """Held-out temporal pattern: random task-local, multi-dimension state changes."""
    base = make_episode(scenario, seed, user, turns, split)
    rng = random.Random(f"renewal-v1/{split}/{seed}/{user}")
    prefs = {}
    for step in base.steps:
        prefs.setdefault(step.observation.task, dict(step.latent))
    for task in TASKS:
        prefs.setdefault(task, {d: rng.choice(v) for d, v in VALUES.items()})
    steps = []
    for step in base.steps:
        obs = step.observation
        changes = []
        if obs.turn > 0 and rng.random() < 0.08:
            task, dim = rng.choice(TASKS), rng.choice(tuple(VALUES))
            value = rng.choice([v for v in VALUES[dim] if v != prefs[task][dim]])
            prefs[task][dim] = value
            changes.append((task, dim, value))
        truth = dict(prefs[obs.task])
        feedback = []
        for e in obs.feedback:
            # Preserve report availability and noise indicator, not the old truth value.
            noisy = e.value != dict(step.target)[e.dim]
            value = rng.choice([v for v in VALUES[e.dim] if v != truth[e.dim]]) if noisy else truth[e.dim]
            feedback.append(Evidence(e.dim, value, e.scope))
        public = Observation(obs.turn, obs.task, tuple(feedback))
        steps.append(Step(public, tuple(truth.items()), tuple(truth.items()), tuple(changes)))
    return replace(base, steps=tuple(steps))


def dataset(scenarios, seeds, users, turns, split):
    return [(renewal_episode if s.name == "ood_renewal" else make_episode)(s, seed, user, turns, split)
            for s in scenarios for seed in seeds for user in range(users)]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_episodes(path, episodes):
    digest = hashlib.sha256()
    with gzip.open(path, "wt", encoding="utf-8") as target:
        for episode in episodes:
            line = json.dumps(episode.to_dict(), sort_keys=True) + "\n"
            digest.update(line.encode())
            target.write(line)
    return digest.hexdigest()


def sources():
    result = provenance()
    for name in ("learned_memory.py", "train_memory_controller.py"):
        result["source_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    result["torch"], result["numpy"] = torch.__version__, np.__version__
    return result


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)
    (path / ".gitignore").write_text("# Keep full local experiment artifacts without committing large files.\n*.gz\n*.npz\n*.pt\nper_user.jsonl\n")


def config_grid():
    return {
        "Context-Latest": [{}],
        "Context-DiscountedCounts": [dict(discount=d, prior=p) for d, p in
                                     itertools.product((0.9, 0.95, 0.98, 0.995), (0.1, 0.5, 1.))],
        "Context-Delta": [dict(eta=e, decay=d) for e, d in
                          itertools.product((0.1, 0.25, 0.4, 0.7), (0.95, 0.99, 1.))],
        "Context-HazardFilter": [dict(hazard=h, noise=n) for h, n in
                                 itertools.product((0.001, 0.005, 0.02, 0.05), (0.05, 0.15, 0.3))],
        "Context-ConstantGate": [dict(write=w, forget=f) for w, f in
                                 itertools.product((0.1, 0.25, 0.5, 0.85), (0., 0.02, 0.1))],
    }


def constant_model(config):
    model = GateController("constant")
    with torch.no_grad():
        values = torch.tensor([config["write"], config["forget"]]).clamp(1e-8, 1 - 1e-8)
        model.logits.copy_(torch.logit(values))
    return model.eval()


def baseline_factory(name, config):
    return {"Context-Latest": Latest, "Context-DiscountedCounts": DiscountedCounts,
            "Context-Delta": DeltaMemory, "Context-HazardFilter": HazardFilter}[name](**config)


@torch.no_grad()
def neural_predictions(model, inputs, batch_size=128, ablation=None):
    collected = {key: [] for key in ("before", "after", "gates")}
    for start in range(0, len(inputs.tasks), batch_size):
        _, trace = rollout(model, inputs.subset(slice(start, start + batch_size)), capture=True, ablation=ablation)
        for key in collected:
            collected[key].append(trace[key])
    return {key: np.concatenate(parts) for key, parts in collected.items()}


def prediction_score(before, targets):
    target = targets.numpy()
    acc = float((before.argmax(-1) == target).mean())
    probs = np.take_along_axis(before, target[..., None], -1)[..., 0]
    return dict(hidden_acc=acc, nll=float(-np.log(np.maximum(probs, 1e-8)).mean()))


def ranking(score):
    return score["hidden_acc"], -score["nll"]


def baseline_score(episodes, name, config):
    correct, nll, total = 0, 0., 0
    for episode in episodes:
        learner = baseline_factory(name, config)
        for step in episode.steps:
            obs = step.observation
            prediction = learner.predict(obs.task, obs.turn)
            hard = labels(prediction)
            for dim, truth in step.latent:
                correct += hard[dim] == truth
                nll -= math.log(max(prediction[dim][VALUES[dim].index(truth)], 1e-8))
                total += 1
            learner.observe(obs)
    return dict(hidden_acc=correct / total, nll=nll / total)


def fit(args):
    metadata = sources()
    new_output(args.output)
    started = time.perf_counter()
    train = dataset(SCENARIOS, TRAIN_SEEDS, args.train_users, args.turns, "controller_train")
    validation = dataset(SCENARIOS, VALIDATION_SEEDS, args.validation_users, args.turns, "controller_validation")
    train_inputs, train_targets = encode_observations(train), encode_targets(train)
    val_inputs, val_targets = encode_observations(validation), encode_targets(validation)
    hashes = {"train": save_episodes(args.output / "train.jsonl.gz", train),
              "validation": save_episodes(args.output / "validation.jsonl.gz", validation)}
    plan = dict(schema="b1.3-controller-v1", supervision="privileged-synthetic-latent-prequential",
                turns=args.turns, train_seeds=TRAIN_SEEDS, validation_seeds=VALIDATION_SEEDS,
                test_seeds=TEST_SEEDS, test_users=args.test_users, train_users=args.train_users,
                validation_users=args.validation_users, initializations=INITIALIZATIONS,
                id_scenarios=[s.name for s in SCENARIOS], ood_scenarios=[s.name for s in OOD_SCENARIOS],
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.003,
                selection="validation macro hidden_acc; NLL breaks ties; one global config per family",
                primary="macro ID hidden_acc vs validation-selected global baseline; practical threshold +0.01",
                source_sha256=metadata["source_sha256"], dataset_sha256=hashes,
                baseline_search={}, controllers=[])
    # Write a design lock before any fit or validation selection; test data are not generated here.
    save_json(args.output / "design.json", plan)
    selected = {}
    for name, configs in config_grid().items():
        candidates = []
        for config in configs:
            if name == "Context-ConstantGate":
                score = prediction_score(neural_predictions(constant_model(config), val_inputs)["before"], val_targets)
            else:
                score = baseline_score(validation, name, config)
            candidates.append(dict(config=config, **score))
        winner = max(candidates, key=ranking)
        selected[name] = winner
        plan["baseline_search"][name] = candidates
        print(f"validation {name}: {winner}", flush=True)
    plan["selected_baselines"] = selected
    plan["reference_baseline"] = max(selected, key=lambda name: ranking(selected[name]))
    for kind, width in (("mlp", 16), ("gru", 8)):
        for seed in INITIALIZATIONS:
            torch.manual_seed(seed)
            model = GateController(kind, width)
            initial = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            shuffle = torch.Generator().manual_seed(seed + 100)
            best, curve = None, []
            training_started = time.perf_counter()
            initial_validation = prediction_score(neural_predictions(model, val_inputs)["before"], val_targets)
            for epoch in range(args.epochs):
                model.train()
                order = torch.randperm(len(train), generator=shuffle)
                losses = []
                for indices in order.split(args.batch_size):
                    optimizer.zero_grad()
                    loss, _ = rollout(model, train_inputs.subset(indices), train_targets[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                score = prediction_score(neural_predictions(model, val_inputs)["before"], val_targets)
                curve.append(dict(epoch=epoch + 1, train_nll=statistics.mean(losses), **score))
                if best is None or ranking(score) > ranking(best):
                    best = dict(epoch=epoch + 1, **score)
                    best_state = copy.deepcopy(model.state_dict())
                print(f"{kind}@{seed} epoch {epoch + 1}: {curve[-1]}", flush=True)
            model.load_state_dict(best_state)
            filename = f"{kind}_{seed}.pt"
            torch.save(dict(kind=kind, width=width, state_dict=best_state), args.output / filename)
            difference = math.sqrt(sum(float(((best_state[k] - initial[k]) ** 2).sum()) for k in initial))
            plan["controllers"].append(dict(kind=kind, width=width, seed=seed, checkpoint=filename,
                                             checkpoint_sha256=hashlib.sha256((args.output / filename).read_bytes()).hexdigest(),
                                             parameters=sum(p.numel() for p in model.parameters()),
                                             parameter_change_l2=difference, initial_validation=initial_validation,
                                             best=best, curve=curve,
                                             fit_seconds=time.perf_counter() - training_started))
    # Families average every initialization; no best-seed cherry-picking on test.
    family_validation = {kind: statistics.mean(c["best"]["hidden_acc"] for c in plan["controllers"] if c["kind"] == kind)
                         for kind in ("mlp", "gru")}
    plan["ablation_family"] = max(family_validation, key=family_validation.get)
    plan["fit_seconds"] = time.perf_counter() - started
    save_json(args.output / "selection.json", plan)
    save_json(args.output / "manifest.json", metadata)
    print(f"Fit locked; reference={plan['reference_baseline']}, ablation family={plan['ablation_family']}", flush=True)


def average_rows(rows, method, scenario=None):
    result = dict(rows[0])
    keys = [key for key, value in rows[0]["metrics"].items() if isinstance(value, (int, float)) or value is None]
    result["method"] = method
    if scenario:
        result["scenario"] = scenario
    result["metrics"] = {key: statistics.mean(r["metrics"][key] for r in rows if r["metrics"][key] is not None)
                         if any(r["metrics"][key] is not None for r in rows) else None for key in keys}
    result["metrics"]["recoveries"] = []  # Event-level records are in individual-run rows, never averaged.
    result["aggregation"] = "arithmetic mean, not prediction ensemble"
    return result


def family_rows(rows, family, names):
    groups = {}
    for row in rows:
        if row["method"] in names:
            groups.setdefault(row["episode"], []).append(row)
    if any(len(group) != len(names) for group in groups.values()):
        raise ValueError("Incomplete initialization pairing")
    return [average_rows(group, family) for group in groups.values()]


def summary_with_reference(rows, reference):
    summary = summarize(rows)
    for scenario, methods in summary.items():
        group = [r for r in rows if r["scenario"] == scenario]
        for method, metrics in methods.items():
            for metric in METRICS:
                differences = []
                for seed in sorted({r["seed"] for r in group}):
                    def seed_values(name):
                        return [r["metrics"][metric] for r in group if r["method"] == name and r["seed"] == seed
                                and r["metrics"][metric] is not None]
                    values, baseline = seed_values(method), seed_values(reference)
                    if values and baseline:
                        differences.append(statistics.mean(values) - statistics.mean(baseline))
                metrics[metric]["delta_vs_selected_baseline"] = interval(differences)
    return summary


def test(args):
    selection_path = args.fit / "selection.json"
    plan = json.loads(selection_path.read_text())
    metadata = sources()
    if metadata["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Experiment sources changed after fit; use a new fit/design lock")
    # Check every checkpoint BEFORE creating or inspecting held-out test data.
    for config in plan["controllers"]:
        path = args.fit / config["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after validation selection")
    new_output(args.output)
    started = time.perf_counter()
    episodes = dataset((*SCENARIOS, *OOD_SCENARIOS), plan["test_seeds"], plan["test_users"], plan["turns"], "controller_test")
    trace_hash = save_episodes(args.output / "episodes.jsonl.gz", episodes)
    inputs = encode_observations(episodes)
    rows = []
    def record(name, episode, result):
        rows.append(dict(episode=episode.key, scenario=episode.scenario.name, seed=episode.seed,
                         user=episode.user, split=episode.split, method=name, metrics=result))
    for name, candidate in plan["selected_baselines"].items():
        if name == "Context-ConstantGate":
            continue
        for episode in episodes:
            record(name, episode, evaluate(episode, baseline_factory(name, candidate["config"])))
        print(f"scored {name}", flush=True)
    models = [("Context-ConstantGate", constant_model(plan["selected_baselines"]["Context-ConstantGate"]["config"]), None)]
    for config in plan["controllers"]:
        saved = torch.load(args.fit / config["checkpoint"], map_location="cpu", weights_only=True)
        model = GateController(saved["kind"], saved["width"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        name = f"Context-{config['kind'].upper()}@{config['seed']}"
        models.append((name, model, None))
        if config["kind"] == plan["ablation_family"]:
            models.append((name + "/no_forget", model, "no_forget"))
            if config["kind"] == "gru":
                models.append((name + "/no_history", model, "no_history"))
    for name, model, ablation in models:
        predictions = neural_predictions(model, inputs, ablation=ablation)
        filename = name.replace("/", "_") + ".npz"
        np.savez_compressed(args.output / filename, **predictions)
        for index, episode in enumerate(episodes):
            adapter = ScoringReplay(tuple(s.observation for s in episode.steps),
                                    predictions["before"][index], predictions["after"][index])
            record(name, episode, evaluate(episode, adapter))
        print(f"scored {name}", flush=True)
    for kind in ("mlp", "gru"):
        name = "Context-" + kind.upper()
        rows.extend(family_rows(rows, name, {f"{name}@{seed}" for seed in plan["initializations"]}))
    ablation_name = "Context-" + plan["ablation_family"].upper()
    for suffix in ("no_forget", "no_history"):
        names = {f"{ablation_name}@{seed}/{suffix}" for seed in plan["initializations"]}
        if any(row["method"] in names for row in rows):
            rows.extend(family_rows(rows, ablation_name + "/" + suffix, names))
    with (args.output / "per_user.jsonl").open("w") as target:
        for row in rows:
            target.write(json.dumps(row, allow_nan=False) + "\n")
    macro = {}
    for row in rows:
        if row["scenario"] in plan["id_scenarios"]:
            macro.setdefault((row["method"], row["seed"], row["user"]), []).append(row)
    macro_rows = [average_rows(group, group[0]["method"], "macro_id") for group in macro.values()]
    summary = summary_with_reference(rows + macro_rows, plan["reference_baseline"])
    save_json(args.output / "summary.json", summary)
    metadata.update(fit_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    selection=plan, test_trace_sha256=trace_hash, episode_keys=[e.key for e in episodes],
                    test_seconds=time.perf_counter() - started,
                    uncertainty="20 data-seed clusters, averaged across 3 training initializations; individual runs retained",
                    interpretation="synthetic mechanism experiment, not online-on-policy or real-user evidence")
    save_json(args.output / "manifest.json", metadata)
    for scenario, methods in summary.items():
        print(f"\n{scenario}", flush=True)
        for name in (*plan["selected_baselines"], "Context-MLP", "Context-GRU"):
            metric = methods[name]["hidden_acc"]
            delta = metric["delta_vs_selected_baseline"]
            print(f"{name:<26} accuracy={metric['estimate']['mean']:.5f} delta={delta['mean']:+.5f} CI={delta['ci95']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    fitting = subparsers.add_parser("fit")
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--epochs", type=int, default=12)
    fitting.add_argument("--train-users", type=int, default=4)
    fitting.add_argument("--validation-users", type=int, default=4)
    fitting.add_argument("--test-users", type=int, default=10)
    fitting.add_argument("--turns", type=int, default=300)
    fitting.add_argument("--batch-size", type=int, default=64)
    testing = subparsers.add_parser("test")
    testing.add_argument("--fit", type=Path, required=True)
    testing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "fit" and (min(args.epochs, args.train_users, args.validation_users, args.test_users, args.batch_size) < 1 or args.turns < 20):
        parser.error("Positive counts and turns >= 20 are required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    (fit if args.stage == "fit" else test)(args)


if __name__ == "__main__":
    main()
