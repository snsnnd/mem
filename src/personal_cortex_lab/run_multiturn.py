"""Step2: Real multi-turn live loop with text utterances.

User subagents produce natural utterances (from their personas, as captured earlier).
Assistant uses TextHebbianSlotRouted pack to style answer in natural language.
User reaction = correction (surprise) or satisfaction based on style mismatch.
We run 10 turns per persona and log slot growth.
"""
from __future__ import annotations
import random, textwrap
from .users import UserPersona
from .assistant import make_answer, correction_event
from .run_text_encoder import UTTERANCES, TextHebbianSlotRouted

# Natural template for assistant replies based on style
def render_answer(style: dict, task: str) -> str:
    fmt = style["format"]; verb = style["verbosity"]; emoji = style["emoji"]
    core = f"任务：{task}"
    body = ""
    if fmt == "code_first":
        body = "```python\ndef quick_sort(a): return sorted(a)\n```\n简短说明：调用 sorted。"
    elif fmt == "table":
        body = "| 项目 | 说明 |\n|---|---|\n| 原理 | 递归划分 |\n| 复杂度 | O(n log n) |"
        if verb == "long": body += "\n详细展开：先选枢轴，再递归，合并是稳定的，适合处理大量数据时注意栈深度。"
    else:  # prose
        body = "这段逻辑是先选一个中间点作枢轴，分成两边依次理顺，整体上是把大的拆小"
        if verb == "long": body += "，最糟 O(n^2)，平均 O(n log n)，原地版不稳定但不占额外内存。"

    if verb == "short" and len(body) > 80 and fmt != "code_first":
        body = body[:60] + "。"

    if emoji == "always":
        body += " ✨ 😊"

    if verb == "short":
        return body.strip()[:160]
    return body.strip()

def run_persona(user: UserPersona, n_turns=10):
    print(f"\n{'='*72}\nUSER {user.uid}  true={user.habit_dict()}\n{'='*72}")
    mem = TextHebbianSlotRouted(eta=0.6, decay=0.99)
    rng = random.Random(123)
    utts = UTTERANCES[user.uid]

    for turn in range(n_turns):
        user_text = rng.choice(utts)
        # Simulate task extracted from user_text (keep same for style demo)
        task = user_text
        pack = mem.get_pack(task)
        style = make_answer(task, pack, rng)
        assistant_text = render_answer(style, task)
        corr = correction_event(style, user.habit_dict())
        # user reaction
        if corr:
            reaction = f"（纠正: {','.join(corr)}）" + rng.choice(utts[:2])
        else:
            reaction = rng.choice(utts[2:3]) if len(utts)>2 else "不错，继续。"
        stats = mem.stats()
        print(f"\n[Turn {turn:02d}] User: {user_text}")
        print(f"           Assistant(pack={pack}): {assistant_text[:90]}")
        print(f"           -> User react: {reaction[:90]} | slots={stats['slots']} growth={stats['growth']}")

        # Update memory with text-feature (task=text for embedding)
        try:
            mem.observe(corr, user.habit_dict(), turn, rng, task)
        except TypeError:
            mem.observe(corr, user.habit_dict(), turn, rng)
    print(f"\nFinal pack: {mem.get_pack()} | Slots: {mem.stats()}")
    return mem

if __name__ == "__main__":
    users = [
        UserPersona(uid="user_01", verbosity="short", emoji="never", fmt="code_first"),
        UserPersona(uid="user_02", verbosity="long", emoji="always", fmt="table"),
        UserPersona(uid="user_03", verbosity="short", emoji="never", fmt="prose"),
    ]
    for u in users:
        run_persona(u, n_turns=10)
    print("\nStep2 done: 3 real personas x 10 turns, slot growth tracked.")
