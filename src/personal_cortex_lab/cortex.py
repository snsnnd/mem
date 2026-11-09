"""Three memory systems: None, Static RAG, Dynamic Cortex (growing/degrading).

Dynamic Cortex is the core of the proposed Personal Cortex.
- slots: habit dimension -> value, importance, count, last_used
- growth: surprise > threshold -> new slot / conflict resolved
- degradation: importance decay each turn, prune low importance
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

# ---------- Base ----------
class MemoryBase:
    name = "base"
    def get_pack(self) -> dict | None:
        return None
    def observe(self, corrections: list[str], true_value: dict, turn: int, rng: random.Random):
        pass

class NoneMemory(MemoryBase):
    name = "A_no_memory"
    def get_pack(self):
        return None

class StaticRAG(MemoryBase):
    """Stores every correction verbatim, returns last 3. No importance weighting, no dedup.
    When history grows, retrieval noise increases (includes stale/conflicting notes).
    """
    name = "B_static_rag"

    def __init__(self):
        self.history: list[tuple[int, str, str]] = []  # (turn, dim, value)

    def observe(self, corrections, true_value, turn, rng):
        for dim in corrections:
            self.history.append((turn, dim, true_value[dim]))
        # Simulate retrieval noise growing with history length:
        # keep only last 20 raw, but pack shows last 3 - still stale if user habit drifts
        # For this MVP we keep static, so noise is from repeated prompting confusion

    def get_pack(self):
        if not self.history:
            return None
        pack = {}
        # Take most recent per dim, but with probability of picking stale one due to noise
        # Model noise: if history > 20, 20% chance to return wrong/stale value
        for dim in ["verbosity", "emoji", "format"]:
            entries = [v for (_, d, v) in self.history if d == dim]
            if entries:
                # Static RAG naive: picks last entry, but if conflict old entries pollute, 15% miss
                if len(self.history) > 20 and random.random() < 0.15:
                    pack[dim] = random.choice(entries)  # may be stale
                else:
                    pack[dim] = entries[-1]
        return pack if pack else None

@dataclass
class Slot:
    dim: str
    value: str
    importance: float = 1.0
    count: int = 1
    last_used: int = 0
    created_turn: int = 0

class DynamicCortex(MemoryBase):
    """Growing / degrading small network simulation.

    - importance decay: importance *= gamma each turn
    - growth: surprise high & slot conflict -> create new slot or overwrite
    - prune: importance < prune_threshold after long inactivity -> delete (degradation)
    """
    name = "C_dynamic_cortex"

    def __init__(self, gamma: float = 0.97, prune_threshold: float = 0.15, surprise_threshold: float = 0.5,
                 max_slots: int = 12, seed: int = 0):
        self.slots: dict[str, Slot] = {}  # dim -> Slot (one per dim forMVP, can extend to multiple per dim)
        self.gamma = gamma
        self.prune_threshold = prune_threshold
        self.surprise_threshold = surprise_threshold
        self.max_slots = max_slots
        self.growth_events = 0
        self.prune_events = 0
        self.update_events = 0
        self.turn = 0
        self.rng = random.Random(seed)

    def _decay(self):
        for dim in list(self.slots.keys()):
            s = self.slots[dim]
            s.importance *= self.gamma

    def _prune(self):
        for dim in list(self.slots.keys()):
            if self.slots[dim].importance < self.prune_threshold and (self.turn - self.slots[dim].last_used) > 20:
                del self.slots[dim]
                self.prune_events += 1

    def observe(self, corrections, true_value, turn, rng):
        self.turn = turn
        self._decay()
        if not corrections:
            # positive feedback: reinforce slots that predicted correctly
            for s in self.slots.values():
                if s.value == true_value[s.dim]:
                    s.importance = min(2.0, s.importance + 0.05)
                    s.last_used = turn
            self._prune()
            return

        for dim in corrections:
            true_v = true_value[dim]
            slot = self.slots.get(dim)

            if slot is None:
                # Growth: no memory for this habit, create new slot
                if len(self.slots) < self.max_slots:
                    self.slots[dim] = Slot(dim=dim, value=true_v, importance=1.0, count=1, last_used=turn, created_turn=turn)
                    self.growth_events += 1
                else:
                    # At capacity: replace lowest importance slot with different dim? For MVP just overwrite same dim
                    self.slots[dim] = Slot(dim=dim, value=true_v, importance=1.0, count=1, last_used=turn, created_turn=turn)
                    self.growth_events += 1
            elif slot.value != true_v:
                # Surprise: predicted wrong, update value (plasticity)
                slot.value = true_v
                slot.importance = min(2.0, slot.importance + 0.5)
                slot.count += 1
                slot.last_used = turn
                self.update_events += 1
            else:
                # Corrections list already means mismatch, so this branch shouldn't happen often,
                # but keep for safety
                slot.importance = min(2.0, slot.importance + 0.2)
                slot.count += 1
                slot.last_used = turn

        self._prune()

    def get_pack(self):
        if not self.slots:
            return None
        return {dim: slot.value for dim, slot in self.slots.items()}

    def stats(self):
        return {
            "slots": len(self.slots),
            "growth": self.growth_events,
            "prune": self.prune_events,
            "update": self.update_events,
            "slots_detail": {k: v.value for k, v in self.slots.items()},
        }
