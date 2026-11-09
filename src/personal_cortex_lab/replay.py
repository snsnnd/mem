"""Candidate-independent, immutable observations for the B1.3 protocol pilot."""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

VALUES = {
    "verbosity": ("short", "long"),
    "emoji": ("never", "always"),
    "format": ("code_first", "table", "prose"),
}
TASKS = ("code", "summary", "explain", "create")


@dataclass(frozen=True)
class Evidence:
    dim: str
    value: str
    scope: str = "unspecified"


@dataclass(frozen=True)
class Observation:
    turn: int
    task: str
    feedback: tuple[Evidence, ...] = ()
    instruction: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Step:
    observation: Observation
    latent: tuple[tuple[str, str], ...]
    target: tuple[tuple[str, str], ...]
    # Evaluator-only: (task, dimension, new persistent value).
    changes: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    feedback_prob: float = 0.3
    noise_prob: float = 0.2
    drift: str = "none"
    temporary_prob: float = 0.0
    scoped: bool = False

    def __post_init__(self):
        for value in (self.feedback_prob, self.noise_prob, self.temporary_prob):
            if not 0 <= value <= 1:
                raise ValueError("Probabilities must be in [0, 1]")
        if self.drift not in ("none", "abrupt", "aba"):
            raise ValueError("Unknown drift mode")


SCENARIOS = (
    Scenario("stationary_clean", feedback_prob=0.3, noise_prob=0),
    Scenario("sparse_noise", drift="abrupt"),
    Scenario("aba", feedback_prob=0.5, noise_prob=0.05, drift="aba"),
    Scenario("transient_scoped", feedback_prob=0.5, noise_prob=0.05,
             temporary_prob=0.15, scoped=True),
    Scenario("transient_unscoped", feedback_prob=0.5, noise_prob=0.05,
             temporary_prob=0.15),
)


@dataclass(frozen=True)
class Episode:
    scenario: Scenario
    seed: int
    user: int
    split: str
    steps: tuple[Step, ...]

    @property
    def key(self):
        return f"{self.split}/{self.scenario.name}/{self.seed}/{self.user}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        steps = []
        for item in data["steps"]:
            obs = item["observation"]
            observation = Observation(
                obs["turn"], obs["task"],
                tuple(Evidence(**e) for e in obs["feedback"]),
                tuple(tuple(pair) for pair in obs["instruction"]),
            )
            steps.append(Step(observation, tuple(map(tuple, item["latent"])),
                              tuple(map(tuple, item["target"])),
                              tuple(map(tuple, item["changes"]))))
        return cls(Scenario(**data["scenario"]), data["seed"], data["user"],
                   data["split"], tuple(steps))


def make_episode(scenario, seed, user, turns=300, split="pilot"):
    if turns < 20:
        raise ValueError("At least 20 turns are required")
    # Scenario name is intentionally absent: matching scenarios can share exogenous draws.
    digest = hashlib.sha256(f"b1.3-v1/{split}/{seed}/{user}".encode()).digest()
    latent_rng = random.Random(int.from_bytes(digest[:8], "big"))
    task_rng = random.Random(int.from_bytes(digest[8:16], "big"))
    report_rng = random.Random(int.from_bytes(digest[16:24], "big"))
    temp_rng = random.Random(int.from_bytes(digest[24:], "big"))
    # No task-name -> format shortcut, and no forced create -> long override.
    base = {task: {d: latent_rng.choice(v) for d, v in VALUES.items()} for task in TASKS}
    first = latent_rng.randint(turns * 2 // 5, turns * 3 // 5)
    second = latent_rng.randint(turns * 7 // 10, turns * 4 // 5)
    steps = []
    for turn in range(turns):
        task = task_rng.choice(TASKS)
        flipped = scenario.drift in ("abrupt", "aba") and turn >= first
        if scenario.drift == "aba" and turn >= second:
            flipped = False
        current = {t: dict(p) for t, p in base.items()}
        if flipped:
            for pref in current.values():
                pref["verbosity"] = "long" if pref["verbosity"] == "short" else "short"
        changes = ()
        if scenario.drift != "none" and (turn == first or
                                         (scenario.drift == "aba" and turn == second)):
            changes = tuple((t, "verbosity", current[t]["verbosity"]) for t in TASKS)
        latent = current[task]
        target = dict(latent)
        temporary = temp_rng.random() < scenario.temporary_prob
        instruction = ()
        if temporary:
            target["verbosity"] = "long" if latent["verbosity"] == "short" else "short"
            if scenario.scoped:
                instruction = (("verbosity", target["verbosity"]),)
        feedback = []
        for dim, options in VALUES.items():
            # Always draw, including when feedback is missing: random streams don't branch.
            present = report_rng.random() < scenario.feedback_prob
            noisy = report_rng.random() < scenario.noise_prob
            wrong = report_rng.choice([v for v in options if v != target[dim]])
            if present:
                scope = "turn" if temporary and dim == "verbosity" and scenario.scoped else "unspecified"
                feedback.append(Evidence(dim, wrong if noisy else target[dim], scope))
        obs = Observation(turn, task, tuple(feedback), instruction)
        steps.append(Step(obs, tuple(latent.items()), tuple(target.items()), changes))
    return Episode(scenario, seed, user, split, tuple(steps))
