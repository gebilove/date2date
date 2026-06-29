# date2date

本项目以“日期格式转换”为一个小型、可控、可自动生成数据的序列转换任务：

```text
YYYY-MM-DD -> DD/MM/YYYY
```

例如：

```text
2026-06-28 -> 28/06/2026
```

项目目标不是用神经网络替代规则字符串处理，而是通过同一个任务，从零实现并比较不同序列建模结构。

本项目从 **Encoder-Decoder RNN** 开始，逐步演进到：

1. Encoder-Decoder RNN
2. Encoder-Decoder RNN + Attention
3. Transformer Encoder-Decoder
4. Decoder-only GPT 风格模型

项目约束：

- 只使用 PyTorch 实现模型、训练循环和评估流程。
- 自己构造训练数据。
- 自己完成训练。
- 自己设计评估指标并分析结果。
- 不使用 Hugging Face Transformers、PyTorch Lightning 或任何预训练模型。

希望通过这个项目观察不同模型在字符级序列转换、位置重排、条件生成和泛化能力上的表现差异。