# Date2Date Transformer 实现 Checkpoints

日期：2026-07-28

目标：在不使用 `torch.nn.Transformer`、Hugging Face、预训练模型的前提下，使用原生 PyTorch 从零实现原始 Encoder-Decoder Transformer，并接入现有 date2date 数据、训练、生成、评估和实验框架。

使用方式：按顺序完成 checkpoint。每次只实现当前组件，通过测试后再继续。第一版以理解原始 Transformer 为目标，不加入 RoPE、FlashAttention、GQA、KV Cache 等现代优化。

---

## 0. 最终要实现多少内容

### 9 个核心类

1. `TokenEmbedding`
2. `SinusoidalPositionalEncoding`
3. `MultiHeadAttention`
4. `PositionwiseFeedForward`
5. `TransformerEncoderLayer`
6. `TransformerEncoder`
7. `TransformerDecoderLayer`
8. `TransformerDecoder`
9. `Date2DateTransformer`

直接使用 PyTorch 的 `nn.Linear`、`nn.LayerNorm`、`nn.Dropout`、`nn.ModuleList`，不为它们再创建包装类。

### 3 个掩码辅助函数

1. `make_src_padding_mask`
2. `make_tgt_padding_mask`
3. `make_causal_mask`

当前 date2date 输入输出均为固定长度，词表也没有 `<PAD>`。因此第一版运行时可以不传 padding mask，但接口必须预留，后续升级为多格式、变长日期时再添加 `<PAD>`。

### 统一符号

```text
B: batch size
S: source sequence length，当前为 10
T: target sequence length，当前训练目标含 EOS，为 11
D: d_model
H: num_heads
Dh: head_dim = D // H
F: d_ff
Vs: source vocab size
Vt: target vocab size
N: encoder/decoder layer count
```

统一采用 `batch_first=True` 风格：

```text
token IDs:       (B, S) 或 (B, T)
hidden states:   (B, S, D) 或 (B, T, D)
attention score: (B, H, Tq, Tk)
logits:          (B, T, Vt)
```

Mask 统一约定：`True` 表示该位置禁止被关注，内部通过 `masked_fill(mask, -inf)` 使用。不要在不同组件中混用相反语义。

### 最终拼装关系

```text
Date2DateTransformer
├── TransformerEncoder
│   ├── TokenEmbedding
│   ├── SinusoidalPositionalEncoding
│   └── TransformerEncoderLayer × N
│       ├── MultiHeadAttention（self-attention）
│       ├── Add + LayerNorm
│       ├── PositionwiseFeedForward
│       └── Add + LayerNorm
├── TransformerDecoder
│   ├── TokenEmbedding
│   ├── SinusoidalPositionalEncoding
│   └── TransformerDecoderLayer × N
│       ├── MultiHeadAttention（masked self-attention）
│       ├── Add + LayerNorm
│       ├── MultiHeadAttention（cross-attention）
│       ├── Add + LayerNorm
│       ├── PositionwiseFeedForward
│       └── Add + LayerNorm
└── output_projection: Linear(D, Vt)
```

---

## Checkpoint 0：确认基线和边界

### 目标

确认现有 RNN/Attention 模型仍能训练和测试，并明确 Transformer 第一版不处理哪些问题。

### 你要完成

- 阅读 `date2date/data.py`，确认 source 长度为 10，target 加 `<EOS>` 后长度为 11。
- 阅读 `date2date/train.py`，确认现有训练接口是 `encoder(input_tensor)` 和 `decoder(encoder_output, target_tensor)`。
- 记录第一版超参数：`d_model=64`、`num_heads=4`、`d_ff=128`、`num_layers=1`、`dropout=0.0`。
- 明确 `d_model % num_heads == 0`。
- 第一版使用原论文的正弦位置编码和 Post-Norm。

### 通过标准

- 能解释为什么 Transformer decoder 的训练输入必须是 `<SOS> + target[:, :-1]`，而监督目标仍是原始 `target`。
- 能解释训练时并行计算全部目标位置，推理时仍逐 token 生成。
- 现有完整测试仍通过。

### 常见问题

- 把 target 原样同时作为 decoder 输入和监督目标，导致当前位置直接看到答案。
- 一开始加入可变长度、RoPE、Pre-Norm 或性能优化，扩大排错范围。

---

## Checkpoint 1：实现 Token Embedding

### 目标

把 token ID 映射为 `D` 维向量，并按原始 Transformer 乘以 `sqrt(D)`。

### `TokenEmbedding` 要实现的接口

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        ...

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        token_ids: (B, L), dtype=torch.long
        返回: (B, L, D)
        """
        ...
```

`__init__` 只创建 embedding 层并保存缩放所需配置；`forward` 负责查表和缩放，不负责位置编码。

### 你要完成

- 使用 `nn.Embedding(vocab_size, d_model)`。
- 检查输入必须是整数 token ID。
- 不在该类中加入位置编码或 dropout。

### 通过标准

- 输入 `(2, 10)`，输出为 `(2, 10, 64)`。
- 输出参与求和并调用 `backward()` 后，embedding 权重存在梯度。
- 能说明为什么乘以 `sqrt(d_model)`。

### 常见问题

- 把 `vocab_size` 和 `d_model` 写反。
- 对 token ID 做浮点转换后再传给 `nn.Embedding`。

---

## Checkpoint 2：实现正弦位置编码

### 目标

为每个序列位置注入固定的绝对位置信息。

### `SinusoidalPositionalEncoding` 要实现的接口

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """
        x: (B, L, D)
        start_pos: 推理增量位置的起点，第一版通常为 0
        返回: (B, L, D)
        """
        ...
```

`__init__` 预计算位置编码并使用 `register_buffer` 保存；`forward` 只截取对应位置、与 `x` 相加并应用 dropout。

### 你要完成

- 偶数维使用 `sin`，奇数维使用 `cos`。
- buffer 建议形状为 `(1, max_len, D)`，便于沿 batch 广播。
- 检查 `start_pos + L <= max_len`。

### 通过标准

- 输入输出 shape 一致。
- 同一个 token embedding 放在不同位置后结果不同。
- `state_dict()` 包含位置编码 buffer，但优化器参数中不包含它。
- `model.to(device)` 后位置编码会一起移动设备。

### 常见问题

- 把位置编码创建成 `nn.Parameter`，导致其被训练。
- 每次 `forward` 都重新生成整张位置编码表。
- 忘记处理奇数 `d_model` 的边界。

---

## Checkpoint 3：实现三种 Mask

### 目标

明确 padding mask 与 causal mask 的 shape、语义和广播规则。

### 要实现的函数接口

```python
def make_src_padding_mask(
    src_tokens: Tensor,
    pad_token_id: int,
) -> Tensor:
    """src_tokens: (B, S)，返回 blocked mask: (B, 1, 1, S)。"""
    ...


def make_tgt_padding_mask(
    tgt_tokens: Tensor,
    pad_token_id: int,
) -> Tensor:
    """tgt_tokens: (B, T)，返回 blocked mask: (B, 1, 1, T)。"""
    ...


def make_causal_mask(
    seq_len: int,
    device: torch.device,
) -> Tensor:
    """返回 blocked mask: (1, 1, T, T)，严格上三角为 True。"""
    ...
```

Decoder self-attention 的最终 mask 使用逻辑或组合：

```text
combined_mask = tgt_padding_mask OR causal_mask
```

### 你要完成

- 先手工写出 `T=4` 的 causal mask，再实现函数。
- 验证 query 位置 0 只能看 key 位置 0。
- 验证 query 位置 3 可以看 key 位置 0 到 3。

### 通过标准

`T=4` 时 blocked mask 的最后两维应等价于：

```text
False True  True  True
False False True  True
False False False True
False False False False
```

### 常见问题

- 上三角和下三角写反。
- 把 `True` 同时当成“可见”和“屏蔽”。
- mask 在 CPU，而 attention score 在其他设备。

---

## Checkpoint 4：实现标准多头注意力

### 目标

实现一个同时支持 self-attention 和 cross-attention 的标准多头注意力类。

### `MultiHeadAttention` 要实现的接口

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        query: (B, Tq, D)
        key: (B, Tk, D)
        value: (B, Tk, D)
        attention_mask: 可广播到 (B, H, Tq, Tk)，True 表示屏蔽
        返回:
          output: (B, Tq, D)
          weights: (B, H, Tq, Tk)
        """
        ...
```

`__init__` 创建 `Wq/Wk/Wv/Wo`、attention dropout，并验证 `d_model % num_heads == 0`；`forward` 完成投影、拆 head、缩放点积、mask、softmax、加权求和、合并 head 和输出投影。

### 你要完成

- 使用 `head_dim = d_model // num_heads`，不是“每个 head 都使用完整 d_model”。
- Q/K/V 投影后变换到 `(B, H, L, Dh)`。
- 缩放因子使用 `sqrt(Dh)`。
- softmax 沿 key 维 `Tk` 计算。
- 第一版显式实现公式，不调用 `nn.MultiheadAttention` 或 SDPA。

### 通过标准

- self-attention：`Tq=Tk=10`，输出 `(B, 10, D)`。
- cross-attention：`Tq=11, Tk=10`，输出 `(B, 11, D)`。
- 每个未完全屏蔽的 attention 行沿 `Tk` 求和约等于 1。
- causal mask 下，所有未来位置权重为 0。
- 输出求和后 `backward()`，四个线性投影都有梯度。

### 常见问题

- 缩放使用 `sqrt(d_model)` 而不是 `sqrt(head_dim)`。
- `transpose` 后直接 `view`，未先保证内存连续或改用 `reshape`。
- mask shape 无法广播到 `(B, H, Tq, Tk)`。
- 当前仓库已有的 RNN 多头注意力维度定义与标准 Transformer 不同，不应直接复制。

---

## Checkpoint 5：实现 Position-wise FFN

### 目标

对每个 token 独立应用同一组两层全连接网络。

### `PositionwiseFeedForward` 要实现的接口

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, D)，返回: (B, L, D)。"""
        ...
```

`__init__` 创建 `D -> F -> D` 的两层线性变换；`forward` 应用第一层、激活、dropout 和第二层。第一版使用 ReLU，与原论文保持一致。

### 通过标准

- 输入输出 shape 相同。
- 改变一个 token 时，不经过 attention 的 FFN 不会直接改变其他 token 的输出。
- 两个线性层均收到梯度。

### 常见问题

- 把序列长度当成线性层输入维度。
- 第二层仍输出 `d_ff`，导致无法做残差相加。

---

## Checkpoint 6：实现 EncoderLayer 和 Encoder

### 目标

把 self-attention、残差、LayerNorm 和 FFN 拼成可堆叠编码器。

### `TransformerEncoderLayer` 要实现的接口

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        x: Tensor,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        x: (B, S, D)
        src_mask: 可广播到 (B, H, S, S)
        返回: 编码结果 (B, S, D)，self-attention 权重 (B, H, S, S)
        """
        ...
```

第一版使用原论文 Post-Norm：

```text
x = norm1(x + dropout(self_attention(x, x, x)))
x = norm2(x + dropout(ffn(x)))
```

### `TransformerEncoder` 要实现的接口

```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        src_tokens: Tensor,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """
        src_tokens: (B, S)
        返回:
          memory: (B, S, D)
          attention_weights: 每层一个 (B, H, S, S)
        """
        ...
```

`__init__` 创建 source embedding、位置编码和 `num_layers` 个独立 EncoderLayer；`forward` 负责 embedding、位置编码和逐层传递。

### 通过标准

- `num_layers=2` 时两层参数对象彼此独立，不是同一个实例的重复引用。
- 输入 `(2, 10)`，memory 输出 `(2, 10, D)`。
- attention 权重列表长度等于 `num_layers`。
- memory 的 loss 反传后，每一层都有梯度。

### 常见问题

- 使用 `[layer] * num_layers`，造成所有层共享同一个模块实例。
- 残差连接两端 shape 不一致。
- LayerNorm 沿错误维度构造；这里应使用 `nn.LayerNorm(d_model)`。

---

## Checkpoint 7：实现 DecoderLayer 和 Decoder

### 目标

实现 masked self-attention、cross-attention 和 FFN，并让 Decoder 兼容现有训练接口。

### `TransformerDecoderLayer` 要实现的接口

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, T, D)
        memory: (B, S, D)
        tgt_mask: 可广播到 (B, H, T, T)
        memory_mask: 可广播到 (B, H, T, S)
        返回:
          hidden: (B, T, D)
          self_weights: (B, H, T, T)
          cross_weights: (B, H, T, S)
        """
        ...
```

Post-Norm 顺序：masked self-attention、cross-attention、FFN；每个子层后都执行残差和 LayerNorm。

### `TransformerDecoder` 要实现的接口

```python
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        sos_token_id: int,
        eos_token_id: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def forward(
        self,
        memory: Tensor,
        target_tensor: Tensor | None = None,
        max_len: int = 20,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        """
        memory: (B, S, D)
        target_tensor: 训练时为 (B, T)，推理时为 None
        返回:
          logits: 训练时 (B, T, Vt)，推理时 (B, generated_len, Vt)
          hidden: None，用于兼容当前 decoder 返回二元组的接口
        """
        ...

    def decode_tokens(
        self,
        decoder_input: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        decoder_input: 已右移的 token IDs，(B, T)
        返回 logits (B, T, Vt) 以及每层 self/cross attention 权重
        """
        ...
```

`__init__` 创建 target embedding、位置编码、DecoderLayer 堆栈和输出投影。`decode_tokens` 只负责一次并行前向。`forward` 负责区分训练和自回归推理。

训练分支必须构造：

```text
decoder_input = concat([SOS], target_tensor[:, :-1])
```

然后一次调用 `decode_tokens`，返回与 `target_tensor` 等长的 logits。

推理分支从 `(B, 1)` 的 SOS 开始，每轮把当前完整前缀传给 `decode_tokens`，取最后位置 logits 的 argmax，并追加到前缀；全部样本生成 EOS 或达到 `max_len` 时停止。

### 通过标准

- 训练输入 target `(B, 11)`，输出 logits `(B, 11, vocab_size)`。
- decoder 输入第一个 token 全为 SOS，后续位置对应 target 的前一项。
- masked self-attention 的未来位置权重为 0。
- cross-attention shape 为 `(B, H, T, S)`。
- 推理不接收真实 target，能够从 SOS 开始生成。

### 常见问题

- cross-attention 的 Q/K/V 来源写错：Q 来自 decoder，K/V 来自 encoder memory。
- 训练时忘记右移 target。
- 将 causal mask 错误应用到 cross-attention；cross-attention 可以看到完整 source。
- 推理时把所有时间位置的 logits 都追加，只应使用最后一个位置。

---

## Checkpoint 8：拼装完整 Date2DateTransformer

### 目标

提供一个能清楚表达完整 Encoder-Decoder Transformer 的顶层类。

### `Date2DateTransformer` 要实现的接口

```python
class Date2DateTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len: int,
        sos_token_id: int,
        eos_token_id: int,
        dropout: float = 0.0,
    ) -> None:
        ...

    def encode(
        self,
        src_tokens: Tensor,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """返回 memory: (B, S, D)。"""
        ...

    def forward(
        self,
        src_tokens: Tensor,
        target_tensor: Tensor | None = None,
        max_len: int = 20,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        """返回训练或生成 logits: (B, T, Vt)。"""
        ...
```

`__init__` 只负责创建 encoder 和 decoder；`forward` 先编码 source，再调用 decoder。

### 你要完成

- 保持顶层类尽量薄，不在这里重复 embedding、mask 或 attention 逻辑。
- 暂不做 source/target embedding 权重共享；date2date 两侧词表相同，但第一版先保持结构清晰。
- 完整模型和 `(encoder, decoder)` 两种调用方式都应可用。

### 通过标准

- `model(src, target)` 输出 `(B, T, vocab_size)`。
- loss 能从输出反传到 encoder embedding 和 decoder embedding。
- `model(src, None, max_len)` 不读取目标答案。

### 常见问题

- 顶层类承担太多细节，导致 encoder/decoder 无法独立测试。
- 只验证 shape，没有确认 encoder 参数实际收到梯度。

---

## Checkpoint 9：接入现有训练与实验框架

### 目标

尽量复用当前实验系统，同时承认 Transformer 与 RNN 的接口差异。

### 你要完成

- 在模型构建逻辑中加入 `transformer` 模型类型。
- 构建后仍返回 `(encoder, decoder)`，以复用现有 `train_batch_samples`、`generate`、`eval_samples`。
- Transformer decoder 返回 `(logits, None)`，保持当前代码的二元组解包方式。
- 更新 `tests/test_models.py`：RNN 模型继续检查 recurrent hidden shape；Transformer 明确检查第二个返回值为 `None`，不要伪造 hidden state。
- 为实验 CLI 增加 `d_ff`、`num_layers` 等必要配置；不要把它们偷偷绑定为 `hidden_size`。
- 第一版 criterion 继续使用 `nn.CrossEntropyLoss()`；当前无 PAD，不需要 `ignore_index`。

### 建议的最小实验配置

```text
d_model=64
num_heads=4
d_ff=128
num_layers=1
dropout=0.0
batch_size=32
train_size=1000
```

### 通过标准

- 单个 batch 的 forward、loss、backward 和 optimizer step 全部成功。
- logits 和 target reshape 后分别为 `(B*T, vocab_size)` 与 `(B*T,)`。
- 实验配置、参数量、loss、预测样例和 checkpoint 能正常保存。

### 常见问题

- 为了兼容 RNN，伪造没有意义的 recurrent hidden state；这里明确返回 `None` 即可。
- encoder 和 decoder 使用不同 `d_model`，导致 cross-attention 无法连接。
- 只修改 `MODEL_NAMES`，忘记更新参数解析、模型构建和测试。

---

## Checkpoint 10：分层验证、过拟合和对比实验

### 目标

证明实现不仅 shape 正确，而且注意力约束、梯度和学习行为都正确。

### 必须覆盖的测试

1. `TokenEmbedding` 输入输出 shape 和梯度。
2. 位置编码 buffer、设备迁移和不同位置差异。
3. causal mask 的精确矩阵。
4. 多头拆分/合并后的 shape。
5. masked attention 的未来权重为 0。
6. EncoderLayer 输入输出 shape 不变。
7. Decoder cross-attention 的 `T × S` shape。
8. target 右移结果准确。
9. 完整模型所有可训练参数均获得梯度。
10. batch size 为 1 和大于 1 都可运行。
11. 训练与自回归推理均返回有限值，不含 NaN/Inf。

### 分阶段训练验证

#### A. 单 batch 过拟合

- 固定 32 条样本。
- 使用 `dropout=0.0`。
- 反复训练同一批数据。
- 目标：训练 exact match 接近 100%。

如果单 batch 无法过拟合，不要扩大数据或模型。优先检查 target shift、mask、softmax 维度和 cross-attention Q/K/V 来源。

#### B. 完整固定格式任务

- 使用现有固定测试集。
- 与 `vanilla_gru`、`bahdanau`、`dot_product_attention`、`multi_head_attention` 对比。
- 记录 exact match、字符准确率、参数量和训练耗时。

#### C. 变长任务前置检查

在引入多格式日期前，再添加 `<PAD>`，启用 padding mask，并将 loss 改为忽略 PAD。不要在固定长度第一版中提前混入这一步。

### 当前阶段最终通过标准

- 9 个核心类和 3 个 mask 函数均有明确测试。
- 能从零画出完整 Encoder-Decoder Transformer 数据流。
- 能说明三种 attention 的 Q/K/V 来源：encoder self-attention、decoder masked self-attention、decoder cross-attention。
- 32 条样本可以过拟合。
- 固定格式日期任务能够完成端到端训练和自回归生成。
- 结果进入现有实验记录，并能与此前 5 个模型公平比较。
- 能指出第一版实现与现代 PyTorch `scaled_dot_product_attention` 的区别。

---

# 推荐执行顺序

```text
Checkpoint 0  基线与边界
    ↓
Checkpoint 1  TokenEmbedding
    ↓
Checkpoint 2  PositionalEncoding
    ↓
Checkpoint 3  Masks
    ↓
Checkpoint 4  MultiHeadAttention
    ↓
Checkpoint 5  PositionwiseFeedForward
    ↓
Checkpoint 6  EncoderLayer + Encoder
    ↓
Checkpoint 7  DecoderLayer + Decoder
    ↓
Checkpoint 8  Date2DateTransformer
    ↓
Checkpoint 9  训练与实验集成
    ↓
Checkpoint 10 测试、过拟合与模型对比
```

# 学习边界

完成上述 checkpoint 后，再进入第二阶段优化：

1. 将显式注意力计算替换为 `torch.nn.functional.scaled_dot_product_attention`，比较数值和性能。
2. 比较 Post-Norm 与 Pre-Norm。
3. 为多格式、变长日期加入 `<PAD>` 和 padding loss mask。
4. 比较正弦位置编码与可学习位置 embedding。
5. 最后再实现 Decoder-only GPT；不要在 Encoder-Decoder Transformer 尚未跑通时混合两种架构。

本阶段的终点不是“写出了 9 个类”，而是这些类能够被逐层验证、完整拼装、接入现有实验框架，并在不读取未来答案的条件下完成 date2date 自回归生成。
