# 如何整理当前 `date2date-seq2seq.ipynb`

日期：2026-06-28

## 背景

当前 notebook 已经完成了一个字符级 seq2seq 的最小闭环：

- 定义字符表：数字、`-`、`/`、`<SOS>`、`<EOS>`；
- 构造日期样本：从 `YYYY-MM-DD` 转成 `DD/MM/YYYY`；
- 实现 `EncoderRNN`；
- 实现 `DecoderRNN`；
- 实现单样本训练函数；
- 实现 `generate` 和 `eval_samples`；
- 跑了一组训练实验。

但当前训练结果显示：

```text
train accuracy: 12 / 1000 0.012
test accuracy: 0 / 100 0.0
```

这说明现在 notebook 虽然代码链路跑通了，但模型还没有真正学会日期格式转换。整理 notebook 的目标不应该只是“把代码排整齐”，而应该把它整理成一个可复现实验记录：能清楚说明当前做到哪一步、发现了什么问题、下一步怎么改。

---

## 建议整理成的 notebook 结构

建议把当前 notebook 重排为下面 9 个部分。

```text
0. 标题与实验目标
1. Imports、随机种子、全局配置
2. 字符表与张量转换工具
3. 数据生成与样本检查
4. Encoder / Decoder 模型定义
5. 训练、生成、评估函数
6. Baseline 实验：当前 vanilla RNN seq2seq
7. 当前结果与问题诊断
8. 下一轮改进方向
9. Checkpoint 总结
```

这样整理后，这个 notebook 不只是“代码草稿”，而是一个可以持续迭代的实验笔记。

---

## 0. 标题与实验目标

开头建议替换掉当前很简略的：

```markdown
# encoder
# decoder
# train
# generate
```

改成：

```markdown
# Date2Date Seq2Seq：日期格式转换实验

目标：训练一个字符级 seq2seq 模型，将日期字符串从：

`YYYY-MM-DD`

转换为：

`DD/MM/YYYY`

例如：

`2026-06-28 -> 28/06/2026`

本 notebook 当前阶段目标：

1. 搭建最小可运行的 Encoder-Decoder RNN；
2. 跑通训练、生成、评估闭环；
3. 记录 baseline 的失败现象；
4. 分析为什么 vanilla RNN 当前效果差；
5. 为下一轮改进做 checkpoint。
```

---

## 1. Imports、随机种子、全局配置

当前 import 比较分散，建议集中到第一段代码里。

推荐整理为：

```python
import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
```

然后单独加一个配置 cell：

```python
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
```

即使暂时不用 GPU，也建议提前加 `DEVICE`，后面 notebook 会更容易扩展。

---

## 2. 字符表与张量转换工具

当前这部分基本可保留：

```python
chars = "0123456789-/"
special_tokens = ["<SOS>", "<EOS>"]

itos = special_tokens + list(chars)
stoi = {ch: i for i, ch in enumerate(itos)}

SOS_token = stoi["<SOS>"]
EOS_token = stoi["<EOS>"]

vocab_size = len(itos)
```

建议补充 markdown 解释：

```markdown
本实验是字符级建模，因此词表中每个字符都是一个 token。

特殊 token：

- `<SOS>`：decoder 生成时的起始 token；
- `<EOS>`：target 序列结束标记。
```

当前 `tensor_from_string` 可以先保留，但建议整理成更明确的版本：

```python
def tensor_from_string(s, add_eos=False):
    ids = [stoi[ch] for ch in s]
    if add_eos:
        ids.append(EOS_token)
    return torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)


def string_from_tensor(tensor):
    chars = []
    tensor = tensor.squeeze(0)

    for token in tensor:
        idx = token.item()
        if idx == EOS_token:
            break
        if idx == SOS_token:
            continue
        chars.append(itos[idx])

    return "".join(chars)
```

重点是：

- `idx = token.item()` 后再访问 `itos[idx]`；
- 所有 tensor 直接放到 `DEVICE`；
- `string_from_tensor` 用 list accumulate，最后 `join`。

---

## 3. 数据生成与样本检查

当前 `make_sample` 里生成的是固定宽度格式：

```python
input = f"{year}-{month:02d}-{day:02d}"
target = f"{day:02d}/{month:02d}/{year}"
```

这很好，建议明确固定任务：

```text
输入长度固定为 10：YYYY-MM-DD
输出长度固定为 10：DD/MM/YYYY
再加 EOS 后 target 长度为 11。
```

但 notebook 里有一些测试样本使用了非固定宽度格式，例如：

```python
"2002-1-23" -> "23/1/2002"
```

这会和数据生成逻辑不一致。

建议统一为固定宽度：

```python
fixed_samples = [
    ("2002-01-23", "23/01/2002"),
    ("1999-12-08", "08/12/1999"),
    ("2020-03-07", "07/03/2020"),
    ("1950-11-28", "28/11/1950"),
    ("2048-06-15", "15/06/2048"),
]
```

这是当前 notebook 需要优先整理的点之一：**训练数据和手写验证样本必须使用同一种格式**。

---

## 4. Encoder / Decoder 模型定义

当前模型结构是：

```text
Encoder:
input token ids
-> embedding
-> RNN
-> final hidden

Decoder:
<SOS>
-> embedding
-> RNN initialized by encoder hidden
-> linear
-> vocab logits
```

建议在模型定义前加一段 markdown：

```markdown
本实验先使用最基础的 Encoder-Decoder RNN，不加 attention。

Encoder 将整个输入日期编码成最后一个 hidden state。
Decoder 使用这个 hidden state 作为初始状态，逐字符生成目标日期。

这个 baseline 的限制是：整个输入序列的信息都被压缩到一个 hidden vector 中，
对于日期格式转换这种位置拷贝任务，可能不够稳定。
```

`EncoderRNN` 里当前还有一些“你来写”的教学注释，如果这个 notebook 作为 checkpoint，建议改成正式注释。

例如：

```python
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_len)
        embedded = self.embedding(input_tensor)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded)
        # output: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        return output, hidden
```

Decoder 目前整体逻辑是对的，但建议把变量命名整理一下：

- `predictIdx` 改成 `pred_token`；
- `input_token` 保持 `(batch_size, 1)`；
- `decoder_outputs` 当前 `torch.stack(outputs, dim=1)` 结果是 `(batch, seq_len, vocab_size)`，这个很好。

---

## 5. 训练、生成、评估函数

当前已经有：

- `train_one_sample`
- `generate`
- `eval_samples`

建议整理顺序为：

```text
5.1 train_one_sample
5.2 generate
5.3 eval_samples
```

### `train_one_sample` 的 checkpoint 建议

当前代码：

```python
input_tensor = tensor_from_string(input_str, True)
target_tensor = tensor_from_string(target_str, True)
```

这里 target 加 EOS 是必要的。

input 是否加 EOS 可以保留，但要在 markdown 里解释清楚：

```markdown
当前实现中，encoder input 也追加 EOS。
这不是必须的；因为输入长度固定，encoder 不依赖 EOS 也能知道序列结束。
为了保持实验简单，后续可以尝试：

- encoder input 不加 EOS；
- decoder target 加 EOS。
```

更推荐下一轮改成：

```python
input_tensor = tensor_from_string(input_str, add_eos=False)
target_tensor = tensor_from_string(target_str, add_eos=True)
```

因为这个任务的输入长度固定，encoder 端没有必要加 EOS。

### `generate` 的 checkpoint 建议

当前 `generate` 中已经调用了：

```python
encoder.eval()
decoder.eval()
```

这是正确的。

建议补充：

```python
encoder.train()
decoder.train()
```

不要写在 `generate` 里面，训练循环里每个 epoch 开始前已经设置 train mode 了。

---

## 6. Baseline 实验：当前 vanilla RNN seq2seq

建议把当前训练代码整理为一个明确的 baseline cell。

推荐 markdown：

```markdown
## Baseline：vanilla RNN Encoder-Decoder

配置：

- train sample size: 1000
- test sample size: 100
- hidden size: 256
- optimizer: Adam
- learning rate: 0.001
- epoch: 100
- teacher forcing: 训练时始终使用 target token 作为 decoder 下一步输入
- decoding: 推理时 greedy decoding
```

训练代码可以先不大改，只把变量集中：

```python
EPOCHS = 100
TRAIN_SIZE = 1000
TEST_SIZE = 100
HIDDEN_SIZE = 256
LR = 0.001
```

然后保留原始结果。

---

## 7. 当前结果与问题诊断

这是最重要的整理部分。当前 notebook 应该明确写出：

```markdown
当前 baseline 没有学会任务。

现象：

- loss 从约 0.63 降到约 0.50；
- train exact-match accuracy 只有 1.2%；
- test exact-match accuracy 是 0%。

这说明模型可能学到了一些局部 token 分布，但没有学会完整字符串转换。
```

### 可能原因 1：exact match 很严格

日期输出必须 10 个字符全部正确才算对。

例如：

```text
target: 28/06/2026
pred:   28/06/2025
```

只错一个字符，exact match accuracy 也是 0。

所以除了完整 accuracy，后续可以增加字符级准确率。

### 可能原因 2：vanilla RNN 没有 attention，位置拷贝困难

这个任务本质是重排字符：

```text
YYYY-MM-DD
DD/MM/YYYY
```

模型需要学会：

```text
input[8:10] -> output[0:2]
input[5:7]  -> output[3:5]
input[0:4]  -> output[6:10]
```

没有 attention 时，decoder 每一步只能依赖 encoder 最后 hidden state，很容易不稳定。

### 可能原因 3：训练方式是单样本 SGD，效率低且噪声大

当前 `train_one_sample` 每次只训练一个样本：

```python
for i, (input_str, target_str) in enumerate(samples):
    loss = train_one_sample(...)
```

这能跑通流程，但训练效率低，也不方便稳定观察 loss。

下一步可以改成 batch training。

### 可能原因 4：样本空间有限但重复采样，没有显式 train/test 切分

日期范围是：

```text
year: 1950-2050，共 101 个
month: 1-12，共 12 个
day: 1-28，共 28 个
总组合：101 * 12 * 28 = 33936
```

训练样本 1000 个是随机采样，有可能重复。测试样本也是随机生成，没有固定 seed 时复现实验困难。

建议后续生成完整样本空间，然后切分 train/test。

---

## 8. 下一轮改进方向

建议下一轮不要一次性改太多，按下面顺序推进。

### 改进 1：统一数据格式

优先级最高。

全部统一为：

```text
input:  YYYY-MM-DD
output: DD/MM/YYYY
```

不要再混用：

```text
2002-1-23
23/1/2002
```

### 改进 2：增加字符级准确率

新增一个函数：

```python
def char_accuracy(samples, encoder, decoder):
    total_chars = 0
    correct_chars = 0

    for input_str, target_str in samples:
        pred = generate(input_str, encoder, decoder, max_len=20)
        max_compare_len = min(len(pred), len(target_str))

        for i in range(max_compare_len):
            total_chars += 1
            if pred[i] == target_str[i]:
                correct_chars += 1

        total_chars += max(0, len(target_str) - max_compare_len)

    return correct_chars / total_chars
```

这样即使 exact match 很低，也能看出模型是否在逐步学会部分结构。

### 改进 3：打印固定样本预测

每隔若干 epoch 打印：

```python
eval_samples(fixed_samples, encoder, decoder, verbose=True)
```

这比只看 loss 更直观。

建议每 10 个 epoch 输出一次固定样本预测。

### 改进 4：尝试 attention

如果目标是学习 seq2seq，下一步最值得做的是加 attention。

原因是这个任务非常适合 attention：decoder 生成 `DD` 时应该关注输入末尾的 day；生成 `MM` 时关注输入中间的 month；生成 `YYYY` 时关注输入开头的 year。

### 改进 5：保留一个 rule-based baseline

这个任务可以用字符串切片 100% 完成：

```python
def rule_based_convert(s):
    year = s[0:4]
    month = s[5:7]
    day = s[8:10]
    return f"{day}/{month}/{year}"
```

建议把它放进 notebook，不是为了替代模型，而是作为任务 sanity check：

```markdown
如果 rule-based baseline 都不能达到 100%，说明数据或评估逻辑有 bug。
模型实验应该和这个 baseline 对齐。
```

---

## 9. Checkpoint 总结

建议在 notebook 最后添加一个 checkpoint markdown cell。

可以写成：

```markdown
## Checkpoint：2026-06-28

当前状态：

- 已完成字符表、样本生成、tensor 转换；
- 已完成 EncoderRNN；
- 已完成 DecoderRNN；
- 已完成单样本训练函数；
- 已完成 greedy generate；
- 已完成 exact-match eval；
- baseline 可以跑通，但效果很差。

当前 baseline 结果：

- train accuracy: 12 / 1000 = 1.2%；
- test accuracy: 0 / 100 = 0%。

当前判断：

- 代码链路基本跑通；
- 但 vanilla RNN encoder-decoder 没有稳定学会日期重排；
- 后续需要统一数据格式、增加评估指标、打印固定样本预测，并优先考虑 attention。

下一步 TODO：

1. 统一所有样本为固定宽度格式；
2. 增加 rule-based baseline；
3. 增加 char-level accuracy；
4. 每 10 epoch 打印 fixed sample 预测；
5. 尝试 encoder input 不加 EOS；
6. 尝试 attention decoder；
7. 如果继续实验，改成 batch training。
```

---

## 推荐的整理顺序

如果现在要实际整理 notebook，建议按这个顺序做：

1. **先不改模型结构**，只重排 notebook 结构；
2. 把 import、config、vocab、data、model、train、eval 分区；
3. 统一固定样本格式；
4. 添加 checkpoint markdown；
5. 添加 rule-based baseline；
6. 添加 char-level accuracy；
7. 重新跑一次 baseline，记录结果；
8. 再开一个新章节做 attention 改进。

不要一上来同时改：

- batching；
- attention；
- GRU/LSTM；
- learning rate；
- 数据集划分；
- 评估指标。

否则如果结果变好或变差，很难知道是哪一个改动导致的。

---

## 当前 notebook 最应该立即修正的 5 个点

### 1. 固定样本格式不一致

当前有：

```python
("2002-1-23", "23/1/2002")
```

但训练样本是：

```python
"2002-01-23" -> "23/01/2002"
```

建议统一成后者。

### 2. 开头 markdown 太粗略

当前：

```markdown
# encoder
# decoder
# train
# generate
```

建议换成实验目标和 checkpoint 说明。

### 3. 没有记录失败分析

当前 notebook 只打印了低准确率，但没有解释为什么低。

建议加“当前结果与问题诊断”章节。

### 4. 只有 exact-match accuracy

建议增加字符级准确率，否则很难判断模型是否在部分学习。

### 5. 没有 rule-based sanity check

日期格式转换可以用切片 100% 完成，建议加 rule-based baseline 确认数据和评估逻辑没问题。

---

## 建议文件命名和版本管理

如果你想保留当前 notebook 的实验痕迹，可以考虑：

```text
date2date-seq2seq.ipynb              # 当前主实验 notebook
docs/answers/organize-date2date-notebook-checkpoint.md  # 本次整理建议
```

如果后续实验变化较大，也可以拆成：

```text
notebooks/01-vanilla-rnn-baseline.ipynb
notebooks/02-rnn-with-attention.ipynb
```

但现在项目还比较小，暂时不一定需要拆文件。优先把当前 notebook 整理清楚即可。

---

## 一句话结论

当前 `date2date-seq2seq.ipynb` 已经完成了 seq2seq 的最小闭环，但 baseline 准确率很低。现在最好的整理方式是把它整理成一个“失败但有价值的 baseline checkpoint”：保留当前结果，明确记录问题，统一数据格式，补充评估指标，然后在下一节有控制地引入 attention 或 batch training。
