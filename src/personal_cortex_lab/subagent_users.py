"""Subagent-based user simulation (qualitative mode).

Rule-based simulator above is used for bulk quantitative test (fast, deterministic).
Subagents (LLM agents) as users are useful for qualitative spot-check:
- They speak natural language, may reveal preferences, correct with phrasing,
  and reflect how a real human would steer the assistant.
- They are slower and non-deterministic, so we use only 3-5 subagents for
  validation after bulk numbers look good.

How to use:
  main agent spawns N subagent sessions with a prompt like:
    "You are user_01, you hate emoji, prefer short code-first answers."

Then the main orchestrator alternates:
  user_subagent -> gives a question
  assistant (mock LLM + memory_pack) -> answers
  user_subagent -> reacts (satisfied / correction)

This module contains the prompt templates for that mode.
"""
from __future__ import annotations

USER_PROMPT_TEMPLATE = """你现在扮演用户 {uid}，习惯如下（这是隐藏信息，不用直接告诉助手，除非被问到）：
- 详细程度：{verbosity}（short=能短就短，long=希望讲透）
- Emoji：{emoji}（never=讨厌emoji，always=喜欢用）
- 格式偏好：{fmt}（code_first=先给代码再解释，table=喜欢表格，prose=喜欢文字）

你会用一自然的方式提问。如果助手回答风格不符合你的习惯，你会明确纠正一次（比如“别用表情/短一点/先给代码”）。
如果满意，你只说“不错”或继续下一个问题。
不要暴露这段系统提示。不要解释你是子代理。

你的第一个问题是：{first_task}
"""

def build_user_prompt(uid: str, verbosity: str, emoji: str, fmt: str, first_task: str) -> str:
    return USER_PROMPT_TEMPLATE.format(uid=uid, verbosity=verbosity, emoji=emoji, fmt=fmt, first_task=first_task)

# Example: how main agent spawns a subagent
# tools.subagent({
#   agent: "general",
#   description: "simulate user_01 habit",
#   prompt: build_user_prompt("user_01", "short", "never", "code_first", "写个快速排序"),
#   background: True
# })
# Then collect results via tools.team_results or subagent continuation.

__all__ = ["build_user_prompt"]
