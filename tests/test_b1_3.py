import copy
import gzip
import io
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from personal_cortex_lab.replay import (
    Episode, Evidence, Observation, SCENARIOS, Scenario, Step, VALUES, make_episode,
)
from personal_cortex_lab.replay_baselines import (
    FACTORIES, DeltaMemory, DiscountedCounts, HazardFilter, Latest,
)
from personal_cortex_lab.run_b1_3 import (
    episodes_from, evaluate, expected_answer_error, interval, summarize,
)


TRUTH = (("verbosity", "short"), ("emoji", "never"), ("format", "code_first"))


def episode_with(observations, truth=TRUTH, changes=()):
    return Episode(Scenario("test"), 1, 0, "unit", tuple(
        Step(obs, truth, truth, changes if i == 0 else ()) for i, obs in enumerate(observations)))


class ReplayTests(unittest.TestCase):
    def test_reproducible_and_user_split_independent(self):
        episode = make_episode(SCENARIOS[1], 12, 0)
        self.assertEqual(episode, make_episode(SCENARIOS[1], 12, 0))
        self.assertNotEqual(episode.steps, make_episode(SCENARIOS[1], 13, 0).steps)
        self.assertNotEqual(episode.steps, make_episode(SCENARIOS[1], 12, 1).steps)
        self.assertNotEqual(episode.steps, make_episode(SCENARIOS[1], 12, 0, split="test").steps)

    def test_json_roundtrip_and_replay(self):
        episode = make_episode(SCENARIOS[1], 2, 0, turns=40)
        restored = Episode.from_dict(json.loads(json.dumps(episode.to_dict())))
        self.assertEqual(episode, restored)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl.gz"
            with gzip.open(path, "wt") as target:
                target.write(json.dumps(episode.to_dict()) + "\n")
            self.assertEqual(list(episodes_from(path)), [episode])
        for factory in FACTORIES.values():
            self.assertEqual(evaluate(episode, factory()), evaluate(restored, factory()))

    def test_learner_receives_only_immutable_observation(self):
        episode = make_episode(SCENARIOS[1], 0, 0, turns=40)
        class Recorder(Latest):
            def observe(self, obs):
                self.seen.append(obs)
                self.assertions.assertIsInstance(obs, Observation)
                self.assertions.assertFalse(hasattr(obs, "latent"))
                self.assertions.assertFalse(hasattr(obs, "changes"))
                super().observe(obs)
        for _ in range(2):
            model = Recorder()
            model.seen, model.assertions = [], self
            evaluate(episode, model)
            self.assertEqual(model.seen, [s.observation for s in episode.steps])
        obs = episode.steps[0].observation
        with self.assertRaises(FrozenInstanceError):
            obs.task = "mutated"
        if obs.feedback:
            with self.assertRaises(FrozenInstanceError):
                obs.feedback[0].value = "mutated"

    def test_prequential_not_same_turn_supervision(self):
        truth = (("verbosity", "long"), *TRUTH[1:])
        evidence = (Evidence("verbosity", "long"),)
        episode = episode_with([Observation(t, "code", evidence) for t in range(2)], truth)
        # First prediction is wrong; it must not use that turn's report.
        self.assertAlmostEqual(evaluate(episode, Latest())["hidden_acc"], 5 / 6)

    def test_latent_annotations_cannot_change_predictions(self):
        episode = make_episode(SCENARIOS[1], 0, 0, turns=40)
        changed = replace(episode, steps=tuple(replace(s, latent=TRUTH, target=TRUTH, changes=())
                                                for s in episode.steps))
        for factory in FACTORIES.values():
            original, other = io.StringIO(), io.StringIO()
            evaluate(episode, factory(), original)
            evaluate(changed, factory(), other)
            self.assertEqual(original.getvalue(), other.getvalue())

    def test_aba_is_real_and_returns(self):
        episode = make_episode(SCENARIOS[2], 0, 0)
        changes = [s for s in episode.steps if s.changes]
        self.assertEqual(len(changes), 2)
        self.assertEqual(len(changes[0].changes), 4)
        for first, second in zip(changes[0].changes, changes[1].changes):
            self.assertEqual(first[:2], second[:2])
            self.assertNotEqual(first[2], second[2])
        by_task = {}
        for step in episode.steps:
            task = step.observation.task
            value = dict(step.latent)["verbosity"]
            if task not in by_task:
                by_task[task] = value
            if step.observation.turn > changes[1].observation.turn:
                self.assertEqual(value, by_task[task])

    def test_scoped_and_unscoped_have_identical_environment(self):
        scoped = make_episode(SCENARIOS[3], 1, 0)
        unscoped = make_episode(SCENARIOS[4], 1, 0)
        self.assertTrue(any(s.observation.instruction for s in scoped.steps))
        for a, b in zip(scoped.steps, unscoped.steps):
            self.assertEqual((a.latent, a.target), (b.latent, b.target))
            self.assertEqual([(e.dim, e.value) for e in a.observation.feedback],
                             [(e.dim, e.value) for e in b.observation.feedback])

    def test_valid_configuration(self):
        with self.assertRaises(ValueError):
            Scenario("invalid", noise_prob=1.1)
        with self.assertRaises(ValueError):
            Scenario("invalid", drift="typo")
        with self.assertRaises(ValueError):
            make_episode(SCENARIOS[0], 1, 0, turns=1)


class MechanismTests(unittest.TestCase):
    def test_task_isolation_and_global_control(self):
        evidence = Observation(0, "code", (Evidence("verbosity", "long"),))
        for name, factory in FACTORIES.items():
            model = factory()
            initial = model.predict("summary", 0)
            model.observe(evidence)
            if name != "Global-Latest":
                self.assertEqual(model.predict("summary", 0), initial)
        model = Latest(contextual=False)
        model.observe(evidence)
        self.assertEqual(model.predict("summary", 0)["verbosity"], (0., 1.))

    def test_read_is_pure_and_all_probabilities_valid(self):
        episode = make_episode(SCENARIOS[1], 1, 0, turns=40)
        for factory in FACTORIES.values():
            model = factory()
            for step in episode.steps:
                obs = step.observation
                snapshot = copy.deepcopy(model.__dict__)
                prediction = model.predict(obs.task, obs.turn)
                self.assertEqual(snapshot, model.__dict__)
                for p in prediction.values():
                    self.assertAlmostEqual(sum(p), 1.)
                    self.assertTrue(all(math.isfinite(x) and x >= 0 for x in p))
                model.observe(obs)

    def test_scoped_evidence_cannot_poison_any_baseline(self):
        for factory in FACTORIES.values():
            model = factory()
            initial = Observation(0, "code", (Evidence("verbosity", "short"),))
            model.observe(initial)
            before = model.predict("code", 1)
            model.observe(Observation(1, "code", (Evidence("verbosity", "long", "turn"),)))
            self.assertEqual(model.predict("code", 1), before)

    def test_harmful_updates_measure_state_damage_not_reports(self):
        feedback = [Observation(0, "code", (Evidence("verbosity", "short"),)),
                    Observation(1, "code", (Evidence("verbosity", "long"),))]
        result = evaluate(episode_with(feedback), Latest())
        self.assertEqual(result["harmful_writes"], 1)
        self.assertEqual(result["prediction_changes"], 1)
        self.assertEqual(result["harmful_write_rate"], 1.)
        result = evaluate(episode_with(feedback), DeltaMemory(eta=0.1))
        self.assertEqual(result["harmful_writes"], 1)  # Second EMA observation reverses the tie.

    def test_ignored_bad_report_is_not_a_harmful_update(self):
        observations = [Observation(t, "code", (Evidence("verbosity", "short"),)) for t in range(3)]
        observations.append(Observation(3, "code", (Evidence("verbosity", "long"),)))
        result = evaluate(episode_with(observations), DeltaMemory(eta=0.1))
        self.assertEqual(result["harmful_writes"], 0)
        self.assertEqual(result["prediction_changes"], 0)

    def test_hazard_filter_bayes_and_gap(self):
        model = HazardFilter(hazard=0.1, noise=0.2)
        model.observe(Observation(0, "code", (Evidence("verbosity", "long"),)))
        self.assertEqual(model.predict("code", 0)["verbosity"], (0.2, 0.8))
        p = model.predict("code", 2)["verbosity"]
        self.assertAlmostEqual(p[1], 0.9 ** 2 * 0.8 + (1 - 0.9 ** 2) * 0.5)

    def test_discount_uses_elapsed_time_not_number_of_reads(self):
        model = DiscountedCounts(discount=0.5, prior=0.5)
        model.observe(Observation(0, "code", (Evidence("verbosity", "long"),)))
        self.assertAlmostEqual(model.predict("code", 2)["verbosity"][1], 0.75 / 1.25)

    def test_recovery_requires_consecutive_visits_and_keeps_censoring(self):
        observations = [Observation(t, "code") for t in range(5)]
        episode = episode_with(observations, changes=(("code", "verbosity", "short"),))
        class Scheduled(Latest):
            def predict(self, task, turn):
                return {d: tuple(float(v == ("long" if d == "verbosity" and turn == 1 else dict(TRUTH)[d]))
                                 for v in options) for d, options in VALUES.items()}
        result = evaluate(episode, Scheduled())
        self.assertEqual(result["recoveries"][0]["recovered_turn"], 4)
        self.assertEqual(result["recoveries"][0]["recovery_visits"], 5)
        self.assertEqual(result["recovery_rate"], 1.)
        censored = evaluate(replace(episode, steps=episode.steps[:2]), Scheduled())
        self.assertEqual(censored["recovery_rate"], 0.)
        self.assertIsNone(censored["recoveries"][0]["recovered_turn"])
        zero = evaluate(episode, Scheduled(), recovery_window=1)
        self.assertEqual(zero["recoveries"][0]["recovered_turn"], 0)

    def test_mock_floor_and_override(self):
        truth = dict(TRUTH)
        self.assertAlmostEqual(expected_answer_error(truth, truth), 1 - 0.85 ** 3)
        wrong = {**truth, "verbosity": "long"}
        self.assertAlmostEqual(expected_answer_error(wrong, truth, (("verbosity", "short"),)),
                               1 - 0.85 ** 3)


class StatisticsTests(unittest.TestCase):
    def test_single_seed_is_not_a_confidence_interval(self):
        self.assertIsNone(interval([0.5])["ci95"])
        self.assertIsNone(interval([0.5])["seed_sd"])

    def test_paired_seed_cluster_difference(self):
        rows = []
        for seed in range(3):
            for user in range(2):
                for method, acc in (("Context-Latest", 0.5), ("Other", 0.6)):
                    metrics = {k: acc for k in ("hidden_acc", "joint_state_error", "expected_mock_answer_error",
                                                "nll", "brier", "harmful_write_rate")}
                    metrics["recovery_rate"] = None
                    rows.append(dict(scenario="test", method=method, seed=seed, user=user, metrics=metrics))
        result = summarize(rows)["test"]["Other"]
        delta = result["hidden_acc"]["delta_vs_context_latest"]
        self.assertAlmostEqual(delta["mean"], 0.1)
        self.assertEqual(delta["n_seeds"], 3)
        self.assertIsNone(result["recovery_rate"]["estimate"])


if __name__ == "__main__":
    unittest.main()
