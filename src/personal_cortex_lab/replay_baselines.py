"""Fixed-capacity mechanism controls; none of these methods is a trained network."""
from __future__ import annotations

from .replay import VALUES


def uniform(dim):
    return tuple(1 / len(VALUES[dim]) for _ in VALUES[dim])


class NoMemory:
    def predict(self, task, turn):
        return {d: uniform(d) for d in VALUES}

    def observe(self, observation):
        pass


class Latest(NoMemory):
    def __init__(self, contextual=True):
        self.contextual = contextual
        self.store = {}

    def _key(self, task):
        return task if self.contextual else "global"

    def predict(self, task, turn):
        stored = self.store.get(self._key(task), {})
        return {d: tuple(float(v == stored[d]) for v in options) if d in stored else uniform(d)
                for d, options in VALUES.items()}

    def observe(self, observation):
        for evidence in observation.feedback:
            if evidence.scope != "turn":
                self.store.setdefault(self._key(observation.task), {})[evidence.dim] = evidence.value


class DiscountedCounts(NoMemory):
    """Discounted Dirichlet counts: heuristic tracking, not an exact dynamic posterior."""
    def __init__(self, discount=0.98, prior=0.5):
        if not 0 < discount <= 1 or prior <= 0:
            raise ValueError("Invalid discount/prior")
        self.discount = discount
        self.prior = prior
        self.counts = {}
        self.last = {}

    def _counts_at(self, task, turn):
        factor = self.discount ** (turn - self.last.get(task, turn))
        stored = self.counts.get(task, {})
        return {d: [factor * x for x in stored.get(d, [0.] * len(v))]
                for d, v in VALUES.items()}

    def predict(self, task, turn):
        counts = self._counts_at(task, turn)
        return {d: tuple((x + self.prior) / (sum(xs) + len(xs) * self.prior) for x in xs)
                for d, xs in counts.items()}

    def observe(self, observation):
        counts = self._counts_at(observation.task, observation.turn)
        for e in observation.feedback:
            if e.scope != "turn":
                counts[e.dim][VALUES[e.dim].index(e.value)] += 1
        self.counts[observation.task] = counts
        self.last[observation.task] = observation.turn


class DeltaMemory(NoMemory):
    """B1.2's so-called Hebbian update is a categorical delta rule / EMA."""
    def __init__(self, eta=0.4, decay=0.99):
        if not 0 < eta <= 1 or not 0 < decay <= 1:
            raise ValueError("Invalid eta/decay")
        self.eta, self.decay = eta, decay
        self.weights = {}

    def predict(self, task, turn):
        stored = self.weights.get(task, {})
        result = {}
        for d in VALUES:
            weights = stored.get(d)
            result[d] = tuple(x / sum(weights) for x in weights) if weights else uniform(d)
        return result

    def observe(self, observation):
        for e in observation.feedback:
            if e.scope == "turn":
                continue
            stored = self.weights.setdefault(observation.task, {})
            old = stored.get(e.dim, [0.] * len(VALUES[e.dim]))
            stored[e.dim] = [self.decay * (1 - self.eta) * x + self.eta * (v == e.value)
                             for x, v in zip(old, VALUES[e.dim])]


class HazardFilter(NoMemory):
    """Finite-state HMM with reset hazard and symmetric noisy reports; NOT BOCPD.

    At each global turn, a slot resets to a uniform latent value with probability h.
    The observation error and hazard are fixed model assumptions, not test metadata.
    """
    def __init__(self, hazard=0.01, noise=0.1):
        if not 0 <= hazard <= 1 or not 0 < noise < 1:
            raise ValueError("Invalid hazard/noise")
        self.hazard, self.noise = hazard, noise
        self.beliefs = {}
        self.last = {}

    def predict(self, task, turn):
        elapsed = turn - self.last.get(task, turn)
        retain = (1 - self.hazard) ** elapsed
        stored = self.beliefs.get(task, {})
        return {d: tuple(retain * x + (1 - retain) / len(options)
                         for x in stored.get(d, uniform(d)))
                for d, options in VALUES.items()}

    def observe(self, observation):
        beliefs = self.predict(observation.task, observation.turn)
        for e in observation.feedback:
            if e.scope == "turn":
                continue
            options = VALUES[e.dim]
            posterior = [p * (1 - self.noise if v == e.value else self.noise / (len(options) - 1))
                         for p, v in zip(beliefs[e.dim], options)]
            beliefs[e.dim] = tuple(p / sum(posterior) for p in posterior)
        self.beliefs[observation.task] = beliefs
        self.last[observation.task] = observation.turn


FACTORIES = {
    "NoMemory": NoMemory,
    "Global-Latest": lambda: Latest(contextual=False),
    "Context-Latest": Latest,
    "Context-DiscountedCounts": DiscountedCounts,
    "Context-Delta": DeltaMemory,
    "Context-HazardFilter": HazardFilter,
}
