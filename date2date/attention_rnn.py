"""Attention-based decoders for date2date sequence conversion."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from attention import AdditiveAttention, DotProductAttention, MultiHeadAttention

try:
    from .data import SOS_token
except ImportError:
    from data import SOS_token


class _DecoderRNNWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        output_size: int,
        attention: nn.Module,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        encoder_output: Tensor,
        target_tensor: Optional[Tensor] = None,
        max_len: int = 20,
    ):
        batch_size = encoder_output.size(0)
        decoder_input = torch.full(
            (batch_size, 1),
            SOS_token,
            dtype=torch.long,
            device=encoder_output.device,
        )
        decoder_hidden = encoder_output[:, -1, :].unsqueeze(0)
        decoder_outputs = []
        steps = target_tensor.size(1) if target_tensor is not None else max_len
        for i in range(steps):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input,
                decoder_hidden,
                encoder_output,
            )
            decoder_outputs.append(decoder_output)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                decoder_input = decoder_output.argmax(dim=-1).detach()

        return torch.cat(decoder_outputs, dim=1), decoder_hidden

    def forward_step(
        self,
        decoder_input: Tensor,
        decoder_hidden: Tensor,
        encoder_output: Tensor,
    ):
        embedded = self.embedding(decoder_input)
        query = decoder_hidden.permute(1, 0, 2)
        context = self.attention(query, encoder_output, encoder_output)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, decoder_hidden)
        return self.out(output), hidden


class DecoderRNN_WithAttention(_DecoderRNNWithAttention):
    """GRU decoder with Bahdanau additive attention."""

    def __init__(self, vocab_size: int, hidden_size: int, output_size: int):
        super().__init__(
            vocab_size,
            hidden_size,
            output_size,
            AdditiveAttention(hidden_size, hidden_size, hidden_size),
        )


class DecoderRNN_WithDotProductAttention(_DecoderRNNWithAttention):
    """GRU decoder with scaled dot-product attention."""

    def __init__(self, vocab_size: int, hidden_size: int, output_size: int):
        super().__init__(
            vocab_size,
            hidden_size,
            output_size,
            DotProductAttention(),
        )


class DecoderRNN_WithMultiHeadAttention(_DecoderRNNWithAttention):
    """GRU decoder with multi-head scaled dot-product attention."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        output_size: int,
        attention_heads: int,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            output_size,
            MultiHeadAttention(
                hidden_size,
                hidden_size,
                hidden_size,
                hidden_size,
                attention_heads,
            ),
        )
