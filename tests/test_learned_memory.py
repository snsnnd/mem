import copy
import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from personal_cortex_lab.learned_memory import (
    GateController, NeuralMemory, PublicBatch, ScoringReplay,
    encode_observations, encode_targets, rollout,
)
from personal_cortex_lab.replay import Episode, Evidence, Observation, SCENARIOS, Scenario, make_episode
from personal_cortex_lab.run_b1_3 import evaluate
from personal_cortex_lab.train_memory_controller import (
    INITIALIZATIONS, TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS,
    average_rows, config_grid, neural_predictions, renewal_episode,
    test as run_held_out_test,
)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models(self):
        for kind, width in (("mlp", 16), ("gru", 8)):
            torch.manual_seed(5)
            model = GateController(kind, width)
            # Nontrivial gates test invariance beyond just the zero-head initialization.
            with torch.no_grad():
                model.head.weight.normal_(std=0.1)
            yield model

    def test_public_encoding_does_not_read_oracle_annotations(self):
        episode = make_episode(SCENARIOS[1], 1, 0, turns=40)
        class PublicOnlyStep:
            def __init__(self, observation):
                self.observation = observation
            @property
            def latent(self):
                raise AssertionError("Oracle read during public encoding")
        public_only = replace(episode, steps=tuple(PublicOnlyStep(s.observation) for s in episode.steps))
        a, b = encode_observations([episode]), encode_observations([public_only])
        self.assertTrue(torch.equal(a.tasks, b.tasks))
        self.assertTrue(torch.equal(a.reports, b.reports))

    def test_gradients_and_optimizer_really_change_parameters(self):
        episodes = [make_episode(SCENARIOS[0], 1, u, turns=40) for u in range(4)]
        inputs, targets = encode_observations(episodes), encode_targets(episodes)
        for model in self.models():
            old = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss, _ = rollout(model, inputs, targets)
            self.assertTrue(loss.requires_grad)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)
            optimizer.step()
            self.assertTrue(any(not torch.equal(old[k], v) for k, v in model.state_dict().items()))

    def test_predictions_do_not_depend_on_targets(self):
        episode = make_episode(SCENARIOS[1], 1, 0, turns=40)
        inputs, targets = encode_observations([episode]), encode_targets([episode])
        for model in self.models():
            _, a = rollout(model, inputs, targets, capture=True)
            altered = targets.clone()
            altered[..., 0] = 1 - altered[..., 0]
            _, b = rollout(model, inputs, altered, capture=True)
            np.testing.assert_array_equal(a["before"], b["before"])
            np.testing.assert_array_equal(a["after"], b["after"])

    def test_future_feedback_cannot_change_prefix_predictions(self):
        inputs = encode_observations([make_episode(SCENARIOS[1], 1, 0, turns=40)])
        altered = PublicBatch(inputs.tasks.clone(), inputs.reports.clone(), inputs.turns.clone())
        altered.reports[:, 20:] = 0
        for model in self.models():
            a = neural_predictions(model, inputs)
            b = neural_predictions(model, altered)
            np.testing.assert_array_equal(a["before"][:, :21], b["before"][:, :21])

    def test_category_permutation_equivariance(self):
        inputs = encode_observations([make_episode(SCENARIOS[1], 2, 0, turns=40)])
        reports = inputs.reports.clone()
        original = reports[..., 0]
        reports[..., 0] = torch.where(original >= 0, 1 - original, original)
        for model in self.models():
            a = neural_predictions(model, inputs)
            b = neural_predictions(model, PublicBatch(inputs.tasks, reports, inputs.turns))
            np.testing.assert_allclose(a["before"][..., 0, :2], b["before"][..., 0, :2][..., ::-1], atol=1e-6)
            np.testing.assert_allclose(a["gates"], b["gates"], atol=1e-6)

    def test_task_renaming_does_not_change_inference(self):
        episode = make_episode(SCENARIOS[1], 1, 0, turns=40)
        renamed = replace(episode, steps=tuple(replace(s, observation=replace(s.observation, task="unseen/" + s.observation.task))
                                                for s in episode.steps))
        for model in self.models():
            a = neural_predictions(model, encode_observations([episode]))
            b = neural_predictions(model, encode_observations([renamed]))
            np.testing.assert_array_equal(a["before"], b["before"])

    def test_live_adapter_batch_and_scoring_adapter_agree(self):
        episodes = [make_episode(SCENARIOS[1], 2, u, turns=40) for u in range(3)]
        inputs = encode_observations(episodes)
        for model in self.models():
            trace = neural_predictions(model, inputs, batch_size=2)
            for index, episode in enumerate(episodes):
                live = NeuralMemory(model)
                for t, step in enumerate(episode.steps):
                    before = live.predict(step.observation.task, t)
                    for d, p in before.items():
                        dim = ("verbosity", "emoji", "format").index(d)
                        np.testing.assert_allclose(p, trace["before"][index, t, dim, :len(p)], atol=1e-6)
                    live.observe(step.observation)
                replay = ScoringReplay(tuple(s.observation for s in episode.steps), trace["before"][index], trace["after"][index])
                a, b = evaluate(episode, NeuralMemory(model)), evaluate(episode, replay)
                for metric in ("hidden_acc", "brier", "nll", "harmful_writes", "recovery_rate"):
                    if a[metric] is not None:
                        self.assertAlmostEqual(a[metric], b[metric], places=6)

    def test_online_updates_never_change_controller_parameters(self):
        episode = make_episode(SCENARIOS[1], 1, 0, turns=40)
        for model in self.models():
            old = copy.deepcopy(model.state_dict())
            evaluate(episode, NeuralMemory(model))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, old[key]))

    def test_missing_or_turn_scoped_report_preserves_all_slot_state(self):
        for model in self.models():
            memory = NeuralMemory(model)
            memory.observe(Observation(0, "new_task", (Evidence("verbosity", "short"),)))
            old = tuple(x.clone() for x in memory.by_task["new_task"])
            memory.observe(Observation(5, "new_task", (Evidence("verbosity", "long", "turn"),)))
            for a, b in zip(old, memory.by_task["new_task"]):
                self.assertTrue(torch.equal(a, b))

    def test_fifth_task_exceeds_fixed_capacity(self):
        memory = NeuralMemory(GateController())
        for i in range(4):
            memory.observe(Observation(i, str(i)))
        with self.assertRaises(ValueError):
            memory.observe(Observation(5, "fifth"))

    def test_checkpoint_roundtrip(self):
        inputs = encode_observations([make_episode(SCENARIOS[1], 1, 0, turns=40)])
        for model in self.models():
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.pt"
                torch.save(model.state_dict(), path)
                restored = GateController(model.kind, model.width)
                restored.load_state_dict(torch.load(path, weights_only=True))
                np.testing.assert_array_equal(neural_predictions(model, inputs)["before"],
                                              neural_predictions(restored, inputs)["before"])

    def test_ablations_and_probabilities_are_finite(self):
        inputs = encode_observations([make_episode(SCENARIOS[1], 1, 0, turns=40)])
        for model in self.models():
            for ablation in (None, "no_forget", "no_history"):
                predictions = neural_predictions(model, inputs, ablation=ablation)
                self.assertTrue(np.isfinite(predictions["before"]).all())
                np.testing.assert_allclose(predictions["before"].sum(-1), 1., atol=1e-6)


class ExperimentDesignTests(unittest.TestCase):
    def test_changed_sources_rejected_before_test_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "selection.json").write_text(json.dumps(dict(source_sha256={"old": "hash"})))
            args = Namespace(fit=root, output=root / "test")
            with patch("personal_cortex_lab.train_memory_controller.sources", return_value={"source_sha256": {"new": "hash"}}), \
                    patch("personal_cortex_lab.train_memory_controller.dataset") as generate:
                with self.assertRaisesRegex(ValueError, "sources changed"):
                    run_held_out_test(args)
                generate.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_changed_checkpoint_rejected_before_test_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(source_sha256={"source": "hash"}, controllers=[dict(checkpoint="model.pt", checkpoint_sha256="invalid")])
            (root / "selection.json").write_text(json.dumps(plan))
            (root / "model.pt").write_bytes(b"changed checkpoint")
            args = Namespace(fit=root, output=root / "test")
            with patch("personal_cortex_lab.train_memory_controller.sources", return_value={"source_sha256": {"source": "hash"}}), \
                    patch("personal_cortex_lab.train_memory_controller.dataset") as generate:
                with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                    run_held_out_test(args)
                generate.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_train_validation_test_seeds_disjoint(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(VALIDATION_SEEDS))
        self.assertFalse(set(TRAIN_SEEDS) & set(TEST_SEEDS))
        self.assertFalse(set(VALIDATION_SEEDS) & set(TEST_SEEDS))
        self.assertEqual(len(INITIALIZATIONS), 3)

    def test_baseline_search_budget_is_recordable(self):
        grid = config_grid()
        for name, configs in grid.items():
            self.assertEqual(len(configs), 1 if name == "Context-Latest" else 12)
            self.assertEqual(len({json.dumps(c, sort_keys=True) for c in configs}), len(configs))

    def test_average_is_not_best_initialization_or_prediction_ensemble(self):
        rows = [dict(method="a", scenario="x", metrics=dict(hidden_acc=v, recovery_rate=None, recoveries=[]))
                for v in (0.6, 0.7, 0.9)]
        averaged = average_rows(rows, "mean")
        self.assertAlmostEqual(averaged["metrics"]["hidden_acc"], 2.2 / 3)
        self.assertIsNone(averaged["metrics"]["recovery_rate"])

    def test_renewal_changes_are_real_task_local_and_multi_dimensional(self):
        scenario = Scenario("ood_renewal", feedback_prob=1., noise_prob=0.)
        episode = renewal_episode(scenario, 3001, 0, 300, "unit")
        self.assertEqual(episode, renewal_episode(scenario, 3001, 0, 300, "unit"))
        changes = [change for step in episode.steps for change in step.changes]
        self.assertGreater(len(changes), 5)
        self.assertGreater(len({c[1] for c in changes}), 1)
        self.assertTrue(all(len(step.changes) <= 1 for step in episode.steps))
        for step in episode.steps:
            for e in step.observation.feedback:
                self.assertEqual(e.value, dict(step.latent)[e.dim])
        self.assertEqual(Episode.from_dict(json.loads(json.dumps(episode.to_dict()))), episode)


if __name__ == "__main__":
    unittest.main()
