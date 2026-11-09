"""User personas simulation.

Each user has a hidden habit vector. The simulator produces requests
and correction feedback based on the gap between assistant output style
and true habits. This replaces a real LLM user - deterministic, fast,
and reproducible for the MVP bulk test.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

VERBOSITY = ["short", "long"]
EMOJI = ["never", "always"]
FORMAT = ["code_first", "table", "prose"]

TASKS = [
    "解释一下你刚才讲的那段代码",
    "用你喜欢的方式讲一遍这道题",
    "帮我写个循环并说明一下",
    "总结一下今天聊的主要内容",
    "再讲一遍刚才的知识点",
    "给我一个例子",
    "把刚才的解释换个格式",
    "介绍一下这个概念",
    "写个快速示例",
    "帮我复述一下要点",
]

@dataclass
class UserPersona:
    uid: str
    verbosity: str  # short | long
    emoji: str      # never | always
    fmt: str        # code_first | table | prose

    def habit_dict(self) -> dict:
        return {"verbosity": self.verbosity, "emoji": self.emoji, "format": self.fmt}

def generate_users(n: int = 30, seed: int = 42) -> list[UserPersona]:
    rng = random.Random(seed)
    users = []
    for i in range(n):
        users.append(UserPersona(
            uid=f"user_{i:02d}",
            verbosity=rng.choice(VERBOSITY),
            emoji=rng.choice(EMOJI),
            fmt=rng.choice(FORMAT),
        ))
    return users

def request_for_task(user: UserPersona, task: str, turn: int) -> str:
    # Simulator generates a user request; style of request hints habit implicitly
    # e.g., code_first users tend to say "先给代码"
    hint = ""
    if user.fmt == "code_first":
        hint = " (先给代码别废话)"
    elif user.fmt == "table":
        hint = " (用表格讲)"
    elif user.verbosity == "short":
        hint = " (短一点)"
    return task + hint
