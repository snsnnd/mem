"""Mock LLM assistant.

In real experiment this would be a call to an LLM API with memory_pack injected.
Here we simulate its behavior: if memory_pack contains correct habit hints,
assistant will with high probability output in matching style.
"""
from __future__ import annotations
import random

def make_answer(task: str, memory_pack: dict | None, rng: random.Random) -> dict:
    """Return assistant answer style attributes.

    memory_pack: dict with keys verbosity/emoji/format or None
    """
    if memory_pack is None:
        # Baseline: noisy generic guess
        return {
            "verbosity": rng.choice(["short", "long"]),
            "emoji": rng.choice(["never", "always"]),
            "format": rng.choice(["code_first", "table", "prose"]),
            "text": f"[generic prose long answer for: {task}]",
            "source": "blind_guess",
        }
    # Has memory: mix of correct recall + hallucination
    # Accuracy depends on how many dims recall correctly
    out = {}
    for dim, options in [("verbosity", ["short", "long"]), ("emoji", ["never", "always"]), ("format", ["code_first", "table", "prose"])]:
        if dim in memory_pack and memory_pack[dim] is not None:
            # 85% follow memory, 15% deviate (LLM imperfect)
            out[dim] = memory_pack[dim] if rng.random() < 0.85 else rng.choice([o for o in options if o != memory_pack[dim]])
        else:
            out[dim] = rng.choice(options)
    out["text"] = f"[answer style={out['verbosity']}/{out['emoji']}/{out['format']} for: {task}]"
    out["source"] = "memory_guided"
    return out

def mismatch_scores(answer: dict, true_habit: dict) -> dict:
    return {
        "verbosity": answer["verbosity"] != true_habit["verbosity"],
        "emoji": answer["emoji"] != true_habit["emoji"],
        "format": answer["format"] != true_habit["format"],
    }

def correction_event(answer: dict, true_habit: dict) -> list[str]:
    """User correction: which dimensions did assistant get wrong?
    User simulator only corrects if mismatch happens."""
    mism = mismatch_scores(answer, true_habit)
    corrections = [k for k, v in mism.items() if v]
    return corrections
