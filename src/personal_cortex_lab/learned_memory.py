"""Shared, category-equivariant write/forget controllers and differentiable replay.

Only public observations enter the cell. Simulator targets belong to the training
loss, not the memory state. At deployment all controller parameters are frozen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .replay import TASKS, VALUES

DIMS = tuple(VALUES)
SIZES = tuple(len(VALUES[d]) for d in DIMS)


@dataclass
class PublicBatch:
    tasks: torch.Tensor       # [episode, turn]
    reports: torch.Tensor     # [episode, turn, dimension], -1 means absent/turn-scoped
    turns: torch.Tensor       # [episode, turn]

    def subset(self, indices):
        return PublicBatch(self.tasks[indices], self.reports[indices], self.turns[indices])


def encode_observations(episodes):
    """Deliberately never reads Step.latent, Step.target, or Step.changes."""
    task_rows, report_rows, turn_rows = [], [], []
    for episode in episodes:
        tasks, reports, turns = [], [], []
        task_to_slot = {}
        for step in episode.steps:
            obs = step.observation
            if obs.task not in task_to_slot:
                if len(task_to_slot) == len(TASKS):
                    raise ValueError("Fixed capacity exceeded")
                task_to_slot[obs.task] = len(task_to_slot)
            tasks.append(task_to_slot[obs.task])
            turns.append(obs.turn)
            row = [-1] * len(DIMS)
            for evidence in obs.feedback:
                if evidence.scope != "turn":
                    row[DIMS.index(evidence.dim)] = VALUES[evidence.dim].index(evidence.value)
            reports.append(row)
        task_rows.append(tasks)
        report_rows.append(reports)
        turn_rows.append(turns)
    return PublicBatch(torch.tensor(task_rows, dtype=torch.long),
                       torch.tensor(report_rows, dtype=torch.long),
                       torch.tensor(turn_rows, dtype=torch.long))


def encode_targets(episodes):
    """Privileged synthetic supervision, used by the offline loss ONLY."""
    return torch.tensor([[[VALUES[d].index(dict(step.latent)[d]) for d in DIMS]
                          for step in episode.steps] for episode in episodes], dtype=torch.long)


class GateController(nn.Module):
    def __init__(self, kind="mlp", width=16):
        super().__init__()
        if kind not in ("mlp", "gru", "constant") or width < 1:
            raise ValueError("Invalid controller configuration")
        self.kind, self.width = kind, width
        self.hidden_size = width if kind == "gru" else 0
        if kind == "constant":
            self.logits = nn.Parameter(torch.tensor([0., -4.]))
        else:
            self.core = nn.GRUCell(6, width) if kind == "gru" else nn.Sequential(nn.Linear(6, width), nn.Tanh())
            self.head = nn.Linear(width, 2)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor([0., -4.]))

    def gates(self, features, hidden):
        if self.kind == "constant":
            return self.logits.sigmoid().expand(*features.shape[:-1], 2), hidden
        flat = features.reshape(-1, 6)
        if self.kind == "gru":
            updated = self.core(flat, hidden.reshape(-1, self.width)).reshape(*features.shape[:-1], self.width)
        else:
            updated = self.core(flat).reshape(*features.shape[:-1], self.width)
        gates = self.head(updated).sigmoid()
        return gates, updated if self.kind == "gru" else hidden

    def update(self, belief, hidden, counts, last, reports, turn, ablation=None):
        """One public observation event per slot. Missing reports are NOT approval."""
        sizes = belief.new_tensor(SIZES)
        mask = torch.arange(3, device=belief.device)[None, :] < sizes[:, None]
        prior = mask.to(belief.dtype) / sizes[:, None]
        present = reports >= 0
        y = torch.nn.functional.one_hot(reports.clamp_min(0), 3).to(belief.dtype)
        entropy = -(belief.clamp_min(1e-12).log() * belief).sum(-1) / sizes.log()
        age = (turn[:, None] - last).clamp_min(0)
        report_probability = torch.where(present, (belief * y).sum(-1), torch.zeros_like(counts))
        features = torch.stack((report_probability, belief.max(-1).values, entropy,
                                counts.log1p() / math.log(301), age.log1p() / math.log(301),
                                (sizes - 2).expand_as(counts)), dim=-1)
        old_hidden = hidden
        if ablation == "no_history":
            hidden = torch.zeros_like(hidden)
        gates, proposed_hidden = self.gates(features, hidden)
        write = gates[..., 0]
        forget = gates[..., 1] if ablation != "no_forget" else torch.zeros_like(write)
        retained = (1 - forget[..., None]) * belief + forget[..., None] * prior
        proposed = (1 - write[..., None]) * retained + write[..., None] * y
        belief = torch.where(present[..., None], proposed, belief)
        hidden = torch.where(present[..., None], proposed_hidden, old_hidden)
        counts = counts + present
        last = torch.where(present, turn[:, None].to(last.dtype), last)
        return belief, hidden, counts, last, gates


def initial_state(model, batch_size):
    parameter = next(model.parameters())
    sizes = parameter.new_tensor(SIZES)
    mask = torch.arange(3, device=parameter.device)[None, :] < sizes[:, None]
    belief = (mask / sizes[:, None]).expand(batch_size, len(TASKS), -1, -1).clone()
    hidden = parameter.new_zeros(batch_size, len(TASKS), len(DIMS), model.hidden_size)
    counts = parameter.new_zeros(batch_size, len(TASKS), len(DIMS))
    last = parameter.new_full(counts.shape, -1.)
    return belief, hidden, counts, last


def rollout(model, inputs, targets=None, capture=False, ablation=None):
    """Full BPTT when targets are given; loss always precedes this turn's update.

    This function never accepts Episodes, scenario metadata, or change labels.
    targets are read exclusively by the loss, never by GateController.update.
    """
    batch_size, turns = inputs.tasks.shape
    state = initial_state(model, batch_size)
    row = torch.arange(batch_size, device=inputs.tasks.device)
    loss = next(model.parameters()).new_zeros(())
    before_rows, after_rows, gate_rows = [], [], []
    for t in range(turns):
        task = inputs.tasks[:, t]
        selected = tuple(x[row, task] for x in state)
        before = selected[0]
        if targets is not None:
            probability = before.gather(-1, targets[:, t, :, None]).squeeze(-1)
            loss = loss - probability.clamp_min(1e-8).log().mean()
        updated = model.update(*selected, inputs.reports[:, t], inputs.turns[:, t], ablation)
        # Out-of-place replacement keeps all previous states valid for BPTT.
        selector = torch.nn.functional.one_hot(task, len(TASKS)).bool()
        next_state = []
        for old, new in zip(state, updated[:4]):
            shape = (batch_size, len(TASKS)) + (1,) * (old.ndim - 2)
            next_state.append(torch.where(selector.reshape(shape), new[:, None], old))
        state = tuple(next_state)
        if capture:
            before_rows.append(before.detach())
            after_rows.append(updated[0].detach())
            gate_rows.append(updated[4].detach())
    trajectory = None
    if capture:
        trajectory = {"before": torch.stack(before_rows, 1).cpu().numpy(),
                      "after": torch.stack(after_rows, 1).cpu().numpy(),
                      "gates": torch.stack(gate_rows, 1).cpu().numpy()}
    return loss / turns, trajectory


class ArrayReplay:
    """Adapter to score previously computed neural traces with the existing evaluator.

    Neural traces are computed from PublicBatch only. Assertions bind every score
    to the same observations; this adapter cannot generate or alter predictions.
    """
    def __init__(self, observations, before, after):
        self.observations = observations
        self.before, self.after = before, after
        self.position, self.updated = 0, False

    def predict(self, task, turn):
        expected = self.observations[self.position]
        if (task, turn) != (expected.task, expected.turn):
            raise ValueError("Neural trace / evaluation observation mismatch")
        array = self.after if self.updated else self.before
        result = {}
        for i, d in enumerate(DIMS):
            p = array[self.position, i, :SIZES[i]].astype(np.float64)
            # float32 round-off must not trip the strict probability check in evaluate().
            result[d] = tuple(p / p.sum())
        return result

    def observe(self, observation):
        if self.updated or observation != self.observations[self.position]:
            raise ValueError("Neural trace / update observation mismatch")
        self.updated = True

    def advance(self):
        # evaluate() performs exactly before -> observe -> after per turn.
        self.position += 1
        self.updated = False


class ScoringReplay(ArrayReplay):
    def predict(self, task, turn):
        result = super().predict(task, turn)
        if self.updated:
            self.advance()
        return result


class NeuralMemory:
    """Live online adapter, including unseen task keys; no training at observe()."""
    def __init__(self, model, ablation=None):
        self.model = model.eval()
        self.ablation = ablation
        self.by_task = {}
        self.last_gates = None

    def _empty(self):
        return tuple(x[:, 0] for x in initial_state(self.model, 1))

    def predict(self, task, turn):
        belief = self.by_task.get(task)
        belief = belief[0] if belief is not None else self._empty()[0]
        result = {}
        for i, d in enumerate(DIMS):
            p = belief[0, i, :SIZES[i]].detach().cpu().double().numpy()
            result[d] = tuple(p / p.sum())
        return result

    @torch.no_grad()
    def observe(self, observation):
        state = self.by_task.get(observation.task)
        if state is None:
            if len(self.by_task) == len(TASKS):
                raise ValueError("Fixed capacity exceeded")
            state = self._empty()
        reports = [-1] * len(DIMS)
        for e in observation.feedback:
            if e.scope != "turn":
                reports[DIMS.index(e.dim)] = VALUES[e.dim].index(e.value)
        device = next(self.model.parameters()).device
        updated = self.model.update(*state, torch.tensor([reports], device=device),
                                    torch.tensor([observation.turn], device=device), self.ablation)
        self.by_task[observation.task] = updated[:4]
        self.last_gates = updated[4].detach().cpu().tolist()
