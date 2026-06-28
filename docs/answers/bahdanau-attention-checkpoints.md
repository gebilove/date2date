# Bahdanau Attention 实现 Checkpoints

日期：2026-06-28

## 背景

当前项目是一个 date2date seq2seq 学习实验，目标是把输入日期字符串转换成目标日期字符串，例如：

```text
输入:  2002-01-23
输出: 23/01/2002
```

当前 notebook 中已经有基础结构：

```text
EncoderRNN
DecoderRNN
train_one_sample
generate
eval_samples
training loop
```

现在希望在现有结构上实现 **Bahdanau Attention**。

核心理解是：

```text
Decoder 每生成一步，都会用当前 decoder hidden 去和 Encoder 的所有 hidden states 做匹配，得到 attention 权重。
```

但要特别注意：

```text
attention 权重是用来加权 encoder hidden states 的，不是用来乘 decoder hidden 的。
```

也就是：

```text
context vector = attention weights × encoder outputs
```

然后再用：

```text
decoder output / hidden + context vector
```

预测当前输出 token。

---

# Bahdanau Attention 实现 Checkpoints

目标：在当前 `date2date-seq2seq.ipynb` 结构上，逐步把普通 Encoder-Decoder RNN 改成带 **Bahdanau Attention** 的 Seq2Seq。

使用方式：按顺序完成 checkpoint。每个 checkpoint 只要求完成当前阶段，不要求一次性做完整实现。

---

## Checkpoint 0：确认当前基线结构

### 目标

先确认当前模型结构和数据流，避免直接加 attention 时不知道该改哪里。

当前结构大致是：

```text
input string
  ↓
tensor_from_string
  ↓
EncoderRNN
  ↓
encoder_output, encoder_hidden
  ↓
DecoderRNN 只使用 encoder_hidden
  ↓
decoder_outputs
  ↓
loss / generate
```

当前问题是：

```text
encoder_output 已经返回了，但 decoder 目前没有使用它。
```

而 Bahdanau Attention 必须使用：

```text
encoder_outputs = [h1, h2, ..., hm]
```

所以后面要改的核心是：

```text
DecoderRNN.forward(...)
从只接收 encoder_hidden
改成同时接收 encoder_outputs 和 encoder_hidden
```

### 你要完成

- 找到当前 notebook 里的：
  - `EncoderRNN`
  - `DecoderRNN`
  - `train_one_sample`
  - `generate`
- 确认 `EncoderRNN.forward()` 返回的是：

```python
encoder_output, encoder_hidden
```

- 打印一次 shape：

```text
encoder_output: (batch_size, input_seq_len, hidden_size)
encoder_hidden: (num_layers, batch_size, hidden_size)
```

### 通过标准

你能解释清楚：

```text
encoder_hidden 是最后一个 hidden，用来初始化 decoder。
encoder_output 是所有时间步的 hidden，用来做 attention。
```

并且当前代码能打印类似：

```text
encoder_output: torch.Size([1, 10, hidden_size])
encoder_hidden: torch.Size([1, 1, hidden_size])
```

### 常见问题

- 不要把 `encoder_hidden` 和 `encoder_output` 混成一个东西。
- `encoder_hidden` 只有最后状态。
- `encoder_output` 才包含每个输入位置的 hidden state。

---

## Checkpoint 1：定义 Attention 的输入输出 shape

### 目标

在写代码前，先固定 Bahdanau Attention 的接口。

推荐在当前项目里实现一个单独的类：

```python
class BahdanauAttention(nn.Module):
    ...
```

它的输入应该是：

```text
decoder_hidden_last: (batch_size, hidden_size)
encoder_outputs:     (batch_size, src_len, hidden_size)
```

输出应该是：

```text
context:      (batch_size, 1, hidden_size)
attn_weights: (batch_size, src_len)
```

### 你要完成

先不要急着接入 Decoder。

只实现 attention 模块，并用假的 tensor 测试：

```text
batch_size = 1
src_len = 10
hidden_size = 32
```

你要验证：

```text
score / energy shape: (batch_size, src_len)
attn_weights shape:   (batch_size, src_len)
context shape:        (batch_size, 1, hidden_size)
```

### 通过标准

你能跑通类似检查：

```python
attn_weights.sum(dim=1)
```

输出应接近：

```text
tensor([1.])
```

并且：

```text
context.shape == (batch_size, 1, hidden_size)
```

你能用自己的话解释：

```text
attn_weights 是对 encoder_outputs 的每个位置分配权重。
context 是 encoder_outputs 的加权求和。
```

### 常见问题

- `attn_weights` 不是 hidden state。
- `attn_weights` 不应该乘 `decoder_hidden`。
- 应该是：

```text
context = attention 权重 × encoder_outputs
```

---

## Checkpoint 2：实现 Bahdanau score 计算

### 目标

实现这个公式对应的代码：

```text
e_ti = v_a^T tanh(W_s s_{t-1} + W_h h_i)
```

其中：

```text
s_{t-1}: decoder 当前 hidden
h_i:     encoder 第 i 个 hidden
e_ti:   第 i 个输入位置的 attention score
```

### 你要完成

在 `BahdanauAttention` 里准备三个线性层：

```text
W_s: hidden_size -> hidden_size
W_h: hidden_size -> hidden_size
v_a: hidden_size -> 1
```

数据流是：

```text
decoder_hidden_last
    ↓
W_s
    ↓
扩展到每个 src_len 位置

encoder_outputs
    ↓
W_h

二者相加
    ↓
tanh
    ↓
v_a
    ↓
energy / score
    ↓
softmax
    ↓
attn_weights
```

### 通过标准

你能打印出：

```text
energy shape:       (batch_size, src_len)
attn_weights shape: (batch_size, src_len)
```

并且：

```python
attn_weights.sum(dim=1)
```

接近：

```text
tensor([1.])
```

### 常见问题

- `softmax` 的维度应该是输入序列长度维度，也就是 `src_len`。
- 不要对 `hidden_size` 维度做 softmax。
- `energy` 可以理解为 attention score，不建议在这里叫 logits。
- 最后的词表预测输出才更适合叫 `vocab logits`。

---

## Checkpoint 3：修改 Decoder 的接口

### 目标

让 Decoder 可以拿到所有 encoder hidden states。

当前 Decoder 大概是：

```python
decoder(encoder_hidden, target_tensor)
```

需要改成：

```python
decoder(encoder_outputs, encoder_hidden, target_tensor)
```

也就是 Decoder 同时使用：

```text
encoder_hidden  → 初始化 decoder hidden
encoder_outputs → 每一步计算 attention
```

### 你要完成

修改：

```python
DecoderRNN.forward(...)
```

让它接收：

```python
def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, max_len=20):
    ...
```

然后每一步调用：

```python
self.forward_step(hidden, input_token, encoder_outputs)
```

也就是：

```python
def forward_step(self, hidden, input_token, encoder_outputs):
    ...
```

### 通过标准

你能跑通一次 teacher forcing：

```python
encoder_output, encoder_hidden = encoder(x)
decoder_outputs, decoder_hidden = decoder(encoder_output, encoder_hidden, y)
```

并得到：

```text
decoder_outputs.shape == (batch_size, target_seq_len, vocab_size)
```

### 常见问题

- 不要再只传 `encoder_hidden`。
- `train_one_sample` 和 `generate` 也都要跟着改。
- 改接口时最容易漏掉 `generate()`。

---

## Checkpoint 4：在 Decoder 单步里加入 context vector

### 目标

把 attention 真正接入 decoder 的每一步生成。

推荐采用当前结构下比较容易理解的一种方式：

```text
1. 用当前 decoder hidden 和 encoder_outputs 算 attention
2. 得到 context
3. 把当前输入 token 的 embedding 和 context 拼接
4. 送进 RNN
5. 用 decoder output / hidden 和 context 一起预测 vocab logits
```

结构可以理解成：

```text
input_token
   ↓
embedding
   ↓
[embedding ; context]
   ↓
decoder RNN
   ↓
decoder output

[decoder output ; context]
   ↓
Linear
   ↓
vocab logits
```

### 你要完成

修改 Decoder 里的几个地方。

#### 1. 增加 attention 模块

```python
self.attention = BahdanauAttention(hidden_size)
```

#### 2. 修改 decoder RNN 输入维度

因为现在 RNN 输入不再只是 embedding，而是：

```text
embedding + context
```

所以输入维度从：

```text
hidden_size
```

变成：

```text
hidden_size * 2
```

也就是：

```python
self.rnn = nn.RNN(hidden_size * 2, hidden_size, batch_first=True)
```

#### 3. 修改输出层输入维度

如果你用：

```text
decoder output + context
```

预测词表，那么 Linear 输入维度也要变成：

```text
hidden_size * 2
```

也就是：

```python
self.out = nn.Linear(hidden_size * 2, output_size)
```

### 通过标准

单步 forward 能打印：

```text
embedded:      (batch_size, 1, hidden_size)
context:       (batch_size, 1, hidden_size)
rnn_input:     (batch_size, 1, hidden_size * 2)
rnn_output:    (batch_size, 1, hidden_size)
logits:        (batch_size, vocab_size)
attn_weights:  (batch_size, src_len)
```

### 常见问题

- `context` 的 shape 建议保持成 `(batch_size, 1, hidden_size)`，方便和 `embedded` 在最后一维拼接。
- `hidden[-1]` 的 shape 是 `(batch_size, hidden_size)`，可以拿来算 attention。
- `output` 通常是 `(batch_size, 1, hidden_size)`。
- 预测 vocab 时不要忘记把 sequence length 这一维处理掉。

---

## Checkpoint 5：让训练流程重新跑通

### 目标

改完 Decoder 后，让 `train_one_sample` 可以重新工作。

原来的训练流程是：

```python
encoder_output, encoder_hidden = encoder(input_tensor)
decoder_outputs, decoder_hidden = decoder(encoder_hidden, target_tensor)
```

现在应该变成：

```python
encoder_outputs, encoder_hidden = encoder(input_tensor)
decoder_outputs, decoder_hidden = decoder(
    encoder_outputs,
    encoder_hidden,
    target_tensor
)
```

### 你要完成

修改：

```python
train_one_sample(...)
```

确保它传入：

```text
encoder_outputs + encoder_hidden
```

然后确认 loss 还能正常计算：

```text
decoder_outputs.reshape(-1, vocab_size)
target_tensor.reshape(-1)
CrossEntropyLoss
```

### 通过标准

跑一个样本训练时，能正常得到 loss：

```text
loss: 一个正常的浮点数
```

例如：

```text
2.x
```

或者训练几步后逐渐下降。

### 常见问题

- `decoder_outputs` 仍然应该是：

```text
(batch_size, target_seq_len, vocab_size)
```

- 如果变成了：

```text
(target_seq_len, batch_size, vocab_size)
```

说明 `stack` 或 `cat` 的维度弄错了。

- `CrossEntropyLoss` 要求输入是：

```text
(N, vocab_size)
```

target 是：

```text
(N,)
```

---

## Checkpoint 6：让 generate 重新跑通

### 目标

训练能跑还不够，推理函数 `generate()` 也要适配 attention。

原来的 generate 是：

```python
encoder_output, encoder_hidden = encoder(input_tensor)
decoder_outputs, decoder_hidden = decoder(encoder_hidden, None, max_len)
```

现在应该变成：

```python
encoder_outputs, encoder_hidden = encoder(input_tensor)
decoder_outputs, decoder_hidden = decoder(
    encoder_outputs,
    encoder_hidden,
    None,
    max_len
)
```

### 你要完成

修改：

```python
generate(...)
```

确保无 teacher forcing 时，Decoder 每一步也能使用 attention。

也就是：

```text
当前预测 token → 下一步输入 token
```

仍然保留，但每一步多了：

```text
hidden + encoder_outputs → attention → context
```

### 通过标准

你能调用：

```python
generate("2002-1-23", encoder, decoder)
```

并且不会报错。

初始结果不要求正确，但必须能生成字符串。

### 常见问题

- 推理阶段没有 `target_tensor`，所以循环长度来自 `max_len`。
- `predictIdx.detach()` 仍然是对的。
- 不要在 `generate()` 里重新训练模型。
- `encoder.eval()` 和 `decoder.eval()` 要保留。

---

## Checkpoint 7：检查 attention 权重是否合理

### 目标

不要只看 loss，要确认 attention 权重真的在对输入位置分配概率。

可以让 Decoder 返回额外的 attention 权重，例如：

```text
decoder_outputs, decoder_hidden, attentions
```

其中：

```text
attentions shape: (batch_size, target_seq_len, src_len)
```

如果暂时不想改返回值，也可以只在 `forward_step` 里临时打印。

### 你要完成

在训练前或训练后，拿一个样本：

```text
input:  2002-1-23
target: 23/1/2002
```

观察每个输出位置对应的 attention 分布。

例如输出第一个字符 `2` 或 `3` 的时候，模型理论上应该更关注输入里的 day 部分。

### 通过标准

你至少能确认：

```text
每个 decoder step 都有一组 attention weights
每组 attention weights 长度等于 input 序列长度
每组 attention weights 的和接近 1
```

### 常见问题

- 刚初始化时 attention 可能接近均匀分布，这是正常的。
- 训练不足时 attention 不一定直观。
- 不要把 attention 权重是否“看起来漂亮”当作唯一指标，最终还是要看生成准确率。

---

## Checkpoint 8：小规模训练验证

### 目标

先不要直接跑大训练，先用极小样本确认模型能过拟合。

推荐使用固定样本：

```python
fixed_samples = [
    ("2002-1-23", "23/1/2002"),
    ("1999-12-8", "8/12/1999"),
    ("2020-3-7", "7/3/2020"),
    ("1950-11-28", "28/11/1950"),
    ("2048-6-15", "15/6/2048"),
]
```

### 你要完成

用这几个样本训练一小段时间，观察：

```text
loss 是否下降
generate 输出是否越来越接近 target
train accuracy 是否上升
```

### 通过标准

在固定小样本上，模型应该能明显过拟合。

理想情况：

```text
train accuracy 接近 1.0
```

至少应该比当前普通 seq2seq 的表现明显更好。

### 常见问题

- 如果小样本都不能过拟合，先不要扩大数据集。
- 优先检查：
  - shape 是否对
  - attention softmax 维度是否对
  - target 是否包含 EOS
  - generate 是否遇到 EOS 就停止
  - `train()` / `eval()` 是否正确切换

---

## Checkpoint 9：再回到随机数据集训练

### 目标

小样本能过拟合后，再回到随机日期转换任务。

当前数据任务是：

```text
输入: YYYY-MM-DD 或类似 2002-1-23
输出: DD/MM/YYYY 或类似 23/1/2002
```

Attention 的作用是让 decoder 每生成一个位置时，可以关注输入中对应的年月日位置。

### 你要完成

重新使用：

```python
make_sample(sample_size)
```

训练并评估：

```text
train accuracy
test accuracy
```

建议先从小配置开始：

```text
sample_size = 100
epoch = 20 或 50
hidden_size = 64 或 128
```

确认能跑通后，再增大。

### 通过标准

至少看到：

```text
loss 下降
train accuracy 明显高于当前基线
test accuracy 有提升
```

如果 test accuracy 仍然很低，先不要急着换复杂模型，先检查数据格式和生成格式是否一致。

### 常见问题

当前 `make_sample()` 用的是补零格式：

```text
2022-01-21 → 21/01/2022
```

但固定样本里有些是不补零的：

```text
2002-1-23 → 23/1/2002
```

这会增加学习难度。

建议之后统一格式，比如都使用：

```text
YYYY-MM-DD → DD/MM/YYYY
```

---

## Checkpoint 10：给 encoder 输入也加 EOS

### 目标

让 encoder 明确知道源序列在哪里结束。

当前改进建议是：

```python
input_tensor = tensor_from_string(input_str, add_eos=True)
```

也就是 encoder 输入也带上 `<EOS>`。

这对 attention 尤其有帮助，因为 encoder_outputs 中会多一个“结束位置”的 hidden state，decoder 可以学习什么时候输入信息已经结束。

### 你要完成

确认以下位置都使用一致的输入方式：

```python
input_tensor = tensor_from_string(input_str, add_eos=True)
```

重点检查：

- `train_one_sample(...)`
- `generate(...)`
- 任意手动测试 cell

### 通过标准

你能解释：

```text
encoder 输入加 EOS 后，encoder_outputs 的 src_len 会比原始字符串长度多 1。
attention weights 的长度也会随之多 1。
```

例如：

```text
input_str 长度: 9
encoder input length: 10
attn_weights length: 10
```

### 常见问题

- 加了 EOS 后，attention 权重长度变化是正常的。
- 不要只在训练时加 EOS，推理时也要一致。
- 如果训练和推理的 encoder 输入格式不一致，模型表现会变差。

---

## Checkpoint 11：整理代码和实验记录

### 目标

当 attention 版本跑通后，把代码整理成清晰结构，并记录实验结果。

### 你要完成

在 notebook 中明确区分：

```text
1. 数据与词表
2. EncoderRNN
3. BahdanauAttention
4. DecoderRNN with Attention
5. train_one_sample
6. generate
7. eval_samples
8. training loop
9. experiment result
```

记录至少这些信息：

```text
hidden_size
sample_size
epoch
learning_rate
train accuracy
test accuracy
是否使用 attention
是否给 encoder 输入加 EOS
观察到的问题
```

### 通过标准

以后回看 notebook 时，能清楚知道：

```text
这个版本有没有 attention
attention 接在 decoder 的哪个位置
训练效果比之前有没有提升
```

### 常见问题

- 不要只改代码不记录结果。
- 不要只看 loss，不看 generate 的实际字符串。
- 不要把普通 decoder 和 attention decoder 混在一起，最好命名区分。

---

# 推荐执行顺序

```text
Checkpoint 0
  ↓
Checkpoint 1
  ↓
Checkpoint 2
  ↓
Checkpoint 3
  ↓
Checkpoint 4
  ↓
Checkpoint 5
  ↓
Checkpoint 6
  ↓
Checkpoint 7
  ↓
Checkpoint 8
  ↓
Checkpoint 9
  ↓
Checkpoint 10
  ↓
Checkpoint 11
```

---

# 当前阶段的最终目标

这一轮的完成标准不是“模型一定达到很高准确率”，而是：

```text
基于当前 EncoderRNN / DecoderRNN 结构，成功接入 Bahdanau Attention。
```

具体来说，完成后应该满足：

- `EncoderRNN` 继续返回：

```text
encoder_outputs, encoder_hidden
```

- `DecoderRNN` 使用：

```text
encoder_hidden 初始化 decoder hidden
encoder_outputs 计算 attention
```

- 每个 decoder step 都能产生：

```text
attention weights
context vector
vocab logits
```

- 训练流程和生成流程都能跑通：

```text
train_one_sample(...)
generate(...)
eval_samples(...)
```

- encoder 输入和推理输入都一致地使用 EOS。
- 你能明确说出：

```text
attention 权重是乘 encoder_outputs，不是乘 decoder hidden。
context vector 再和 decoder hidden/output 一起用于预测当前输出 token。
```

最关键的实现路线可以记成一句话：

```text
先让 Decoder 拿到 encoder_outputs，再在 forward_step 里用 hidden + encoder_outputs 算 attention，得到 context，最后用 context + decoder output 预测词表。
```

---

# 建议的下一步

建议下一步只做两件事：

1. 先单独实现 `BahdanauAttention`，用假 tensor 验证 shape。
2. 再改 `DecoderRNN.forward` 和 `forward_step`，不要一开始就改训练循环。

这样如果出错，基本可以很快定位是在 attention 模块、decoder 接口，还是训练流程。