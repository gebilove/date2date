# date2date seq2seq 真正训练 Checkpoints

目标：基于 `date2date-seq2seq.ipynb` 里已经写好的 Encoder、Decoder、样本生成、单样本训练和 generate 框架，逐步补齐“真正训练”的能力。

原则：这里不直接给完整实现代码，只定义每个 checkpoint 你要完成什么、如何自测、什么现象说明通过。

---

## Checkpoint 0：确认当前最小链路可运行

### 目标
确认 notebook 当前已经能完成一次完整前向和一次单样本反向传播。

### 你要检查
- `EncoderRNN` 能接收 shape 为 `(1, input_seq_len)` 的输入。
- `DecoderRNN` 能接收 `encoder_hidden` 和 `target_tensor`。
- `train_one_sample(...)` 能返回一个 loss 数值。
- `generate(...)` 能运行，不一定输出正确结果。

### 通过标准
运行单样本训练时没有报错，并且能看到类似：

```text
loss = 2.x
```

### 注意
如果 `DecoderRNN.forward` 仍然出现：

```text
SyntaxError: non-default argument follows default argument
```

说明函数参数顺序或默认值还没修正。你需要先保证 `DecoderRNN` 类定义能被 notebook 正常执行。

---

## Checkpoint 1：把样本生成改成可控规模

### 目标
让 `make_sample` 可以生成任意数量的数据，而不是固定 100 条。

### 你要完成
- 给 `make_sample` 增加一个参数，例如 `n`。
- 调用时可以生成：
  - 训练集：几千到一万条。
  - 测试集：几百到一千条。

### 通过标准
你可以写出类似这样的调用方式：

```python
train_inputs, train_targets = make_sample(...)
test_inputs, test_targets = make_sample(...)
```

并确认：

```python
len(train_inputs) == len(train_targets)
len(test_inputs) == len(test_targets)
```

### 不要急着做
这一阶段不要做 batch、padding、DataLoader。

---

## Checkpoint 2：清理单样本训练函数

### 目标
让 `train_one_sample` 成为一个安静、稳定、可被循环反复调用的训练单元。

### 你要完成
- 保留它的核心流程：
  1. 字符串转 tensor。
  2. encoder 前向。
  3. decoder teacher forcing 前向。
  4. reshape logits 和 target。
  5. 计算 CrossEntropyLoss。
  6. backward。
  7. optimizer step。
  8. 返回 `loss.item()`。
- 去掉训练过程里的调试打印，例如：

```python
print("logits", logits.shape)
```

### 通过标准
连续调用多次 `train_one_sample(...)` 不刷屏、不报错，每次都返回 loss 数值。

---

## Checkpoint 3：实现一个 epoch 内的训练循环

### 目标
从“训练一个样本一次”升级到“遍历整个训练集一次”。

### 你要完成
定义一个训练函数或代码块，完成：

- 把 `train_inputs` 和 `train_targets` 配对。
- 每个 epoch 开始前打乱样本顺序。
- 对每一对 `(input_str, target_str)` 调用 `train_one_sample`。
- 累加 loss。
- 每隔一定 step 输出平均 loss。

### 通过标准
你能看到训练过程持续输出，例如：

```text
step=1000, avg_loss=...
step=2000, avg_loss=...
```

并且程序能完整跑完一个 epoch。

### 观察重点
不要只看某一次 loss，要看平均 loss 是否整体下降。

---

## Checkpoint 4：实现多 epoch 训练

### 目标
让模型真正反复学习数据分布，而不是只看一遍训练集。

### 你要完成
- 在 Checkpoint 3 的基础上增加 epoch 外层循环。
- 每个 epoch 结束时输出一次汇总信息。
- 初始建议：
  - `hidden_size` 从 64 或 128 开始。
  - `lr` 从 0.001 开始。
  - `epochs` 从 5 到 20 之间试。

### 通过标准
训练日志中平均 loss 应该有明显下降趋势。

例如从接近：

```text
2.x
```

逐渐下降到更低。

### 注意
如果 loss 长时间不降，优先排查：

- target 是否加了 `<EOS>`。
- decoder 输出长度是否等于 target 长度。
- logits reshape 后是否是 `(-1, vocab_size)`。
- target reshape 后是否是 `(-1,)`。
- 学习率是否过大。

---

## Checkpoint 5：用 generate 做定性测试

### 目标
训练后用几个固定输入检查模型是否学会了日期重排。

### 你要完成
准备一组固定样例，例如：

```text
2002-1-23
1999-12-8
2020-3-7
1950-11-28
2048-6-15
```

训练后调用 `generate`，观察输出。

### 通过标准
输出应逐渐接近：

```text
23/1/2002
8/12/1999
7/3/2020
28/11/1950
15/6/2048
```

### 观察重点
早期模型可能只学会部分模式，例如：

- 能输出 `/`，但数字顺序不对。
- 年份能输出，日/月不稳定。
- 输出长度不稳定。
- 忘记停止，没有生成 `<EOS>`。

这些都可以作为后续调参依据。

---

## Checkpoint 6：实现 exact match accuracy

### 目标
不要只靠肉眼看样例，要用测试集准确率评估模型是否真的学会。

### 你要完成
实现一个评估函数，逻辑是：

- 遍历测试集。
- 对每个 `input_str` 调用 `generate`。
- 将预测字符串和 `target_str` 做完全相等比较。
- 统计正确数量和准确率。

### 通过标准
能输出类似：

```text
accuracy: 73/200 = 0.3650
```

### 观察重点
这是严格指标。哪怕只错一个字符，也算错。

例如：

```text
预测: 23/1/2020
目标: 23/1/2002
```

算错。

---

## Checkpoint 7：做第一轮超参数实验

### 目标
找到当前 RNN 框架下比较容易收敛的设置。

### 你要尝试
至少比较几组：

| hidden_size | lr | 观察 |
|---|---:|---|
| 32 | 0.001 | 是否能下降 |
| 64 | 0.001 | 是否更稳定 |
| 128 | 0.001 | 是否更容易记住完整日期 |
| 128 | 0.01 | 是否震荡或不稳定 |

### 通过标准
你能记录每组的大致现象：

- loss 下降速度。
- generate 样例质量。
- exact match accuracy。

### 注意
同一个任务不要频繁同时改太多东西。一次只改一个关键变量，方便判断原因。

---

## Checkpoint 8：可选升级 RNN 为 GRU

### 目标
如果普通 `nn.RNN` 学得慢或效果差，尝试用 `nn.GRU` 替换。

### 你要完成
- Encoder 里的 `nn.RNN` 替换为 `nn.GRU`。
- Decoder 里的 `nn.RNN` 替换为 `nn.GRU`。
- 尽量不改其他训练逻辑。

### 通过标准
替换后：

- shape 仍然对齐。
- `train_one_sample` 能正常训练。
- `generate` 能正常生成。
- loss 或 accuracy 相比普通 RNN 有改善。

### 为什么这个 checkpoint 放后面
先用普通 RNN 跑通完整训练闭环，再换 GRU。这样你能清楚知道性能提升来自模型结构，而不是来自其他修复。

---

## Checkpoint 9：暂缓 batch 训练

### 目标
明确当前阶段不优先做 batch，避免一次引入太多复杂度。

### 原因
日期字符串长度不一致：

```text
2002-1-3      length 8
2002-12-23    length 10
```

如果 batch 训练，需要额外处理：

- padding。
- `<PAD>` token。
- `ignore_index`。
- 输入和目标的 batch padding。
- loss mask。

### 当前通过标准
只要单样本循环训练能让模型学会任务，就先不进入 batch。

---

## Checkpoint 10：整理实验记录

### 目标
每次训练后留下可比较的记录，而不是只看当次 notebook 输出。

### 你要记录
建议记录：

```text
日期：
模型：RNN / GRU
hidden_size：
lr：
训练样本数：
测试样本数：
epochs：
最终 avg loss：
测试 accuracy：
典型错误：
结论：
```

### 通过标准
你能根据记录回答：

- 当前哪个配置最好？
- 模型主要错在哪里？
- 下一步应该调学习率、hidden_size，还是换 GRU？

---

# 推荐执行顺序

```text
Checkpoint 0
  -> Checkpoint 1
  -> Checkpoint 2
  -> Checkpoint 3
  -> Checkpoint 4
  -> Checkpoint 5
  -> Checkpoint 6
  -> Checkpoint 7
  -> Checkpoint 8 可选
  -> Checkpoint 9 以后再做
  -> Checkpoint 10 持续记录
```

# 当前阶段的最终目标

在不做 batch、不加 attention 的前提下，先达成：

```text
输入：yyyy-m-d
输出：d/m/yyyy
```

并且在测试集上能用 exact match accuracy 证明模型不是只记住了单个样本，而是学到了日期格式转换规律。
