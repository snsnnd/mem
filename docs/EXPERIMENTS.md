# EXPERIMENTS

所有实验都在 `uv run python -m personal_cortex_lab.run_*` 里可复现。

## 1. 静态习惯 (30x80)

目的：验证记忆本身有用。

```
A-无记忆  corr 0.92 recall 0%
B-静态RAG corr 0.402 recall 100%
C-动态    corr 0.402
D-全局MLP corr 0.407
E-槽位MLP corr 0.405
F-Hybrid  corr 0.402
G-Hebbian corr 0.403
```

结论：静态下全同，差距在漂移。

## 2. 漂移 (40 轮翻转)

```
B 0.490
C 0.417
D 0.476 (weight_decay 0.01->0.05 只降到 0.476)
E 0.526
F 0.417
G Hebbian fast 0.407
```

结论：梯度慢，赫布快版追到规则。快显式跳变 + 慢管理最稳。

## 3. 槽位MLP vs 全局MLP

* 静态都 0.405 左右
* 漂移 SlotMLP 0.524，Global 0.476，规则 0.417

结论：分区单独解决不了稳定性-可塑性冲突，值也用梯度学太慢。

## 4. Hybrid (显式值 + MLP 管理)

```
静态 0.402
漂移 0.417
```

与规则同分，且加了个可学习的剪枝层。

## 5. 按任务分片细节

4 任务：code/summary/explain/create，0-30 code+summary，30-60 explain+create，60-100 code+summary 回来。

```
B 单值槽    overall 0.636 (code 0.69 summary 0.66...)
H 分片      overall 0.406 (code 0.40 summary 0.38...)
```

槽位重要性轨迹：能看到 code 闲置时从 2.0 降到 0.52，回来又被 reinforcement 到 1.95。剪枝在 `gamma=0.90` 时 50 轮触发 prune=1。
结论：分片主导 22 个点的差距。

## 6. 真人话子代理 (真人 utterances)

3 子代理人格（user_01/02/03）。

```
user_01 (short/never/code_first) -> 最终 pack 100% 对
user_02 (long/always/table) -> 100% 对
user_03 (short/never/prose) -> 100% 对
```

哈希编码器只是翻译官代理，链路仍然成立。

## 7. 10 轮真多轮

每个用户 10 轮自然语言多轮，槽涨 2 次后就稳了，偶尔触发一次生长（两次纠正都会带新格子）。

## 8. Prompt vs ΔW (Hypernet)

```
Prompt pack  corr 0.404  (~+30 tokens)
Hypernet     corr 0.182  (0 tokens, weight merge)
```

Hypernet 权重生成比文字包更准确、更省 token。

## 9. 真 Qwen2.5-0.5B (4060 FP16)

* Base 1GB，显存 ~1.2GB
* LoRA per-habit 30 步蒸馏：

```
code_first loss 2.24->1.02 norm 2.40
table      loss 1.58->0.59 norm 2.38
prose      loss 4.25->2.06 norm 2.38
```

* Hypernet 学 `habit_vec -> ideal ΔW`：`MSE 0.0085 -> 0.0000`

零样本组合 `long+emoji+code_first` 生成了代码，但没生成 emoji，因为训练基里没见过带 emoji 的习惯。预计是补一轮 emoji 数据就能泛化，训练集覆盖度决定泛化上限。

---

## 关键结论速览

* 预测是核心：惊讶 = 预测错的维度
* 快系统（显式跳变）决定下限，慢系统（MLP/Hebbian）决定上限
* 分区必须带任务键，否则上下文干扰不可分离
* 权重注入比文字注入快一倍且省 token
