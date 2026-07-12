"""vanilla Encoder-Decoder RNN（不加 attention）。"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

try:
    from .data import SOS_token
except ImportError:
    from data import SOS_token


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.0)

    # input.shape (batch_size, seqlen)
    def forward(self, input):
        # 要限定 input的shape
        input = self.embedding(input)
        input = self.dropout(input)
        # hidden不受 batch_first=True的控制，其shape仍然为(direction*num,batch_size,hidden_size)
        # output.shape 是 (batch_size, seq_len, direction*hidden_size)
        output, hidden = self.rnn(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    # encoder_output.shape: (batch_size, source_seq_len, hidden_size)
    # target_tensor.shape: (batch_size, target_seq_len)
    # batch_size: 一个批次中的日期样本数量
    # source_seq_len: 每条输入序列的 token 数量，补齐后通常相同
    # hidden_size: RNN 在每个时间步输出的隐藏状态特征维度
    def forward(self, encoder_output: Tensor, target_tensor: Optional[Tensor] = None, max_len: int = 20):
        batch_size = encoder_output.shape[0]
        hidden = encoder_output[:, -1: , :].permute(1, 0, 2)
        outputs = []
        # input_token的格式是 (batch_size, seq_len)
        input_token = torch.full(
            (batch_size, 1), SOS_token, dtype=torch.long, device=encoder_output.device
        )  # 设置起始词元
        loop_len = max_len
        if target_tensor is not None:
            loop_len = target_tensor.shape[1]

        # 循环时间步
        for i in range(loop_len):
            logits, hidden, pred_token = self.forward_step(input_token , hidden)
            outputs.append(logits)

            if target_tensor is not None:
                input_token = target_tensor[:,i].unsqueeze(1)
            else :
                input_token = pred_token.detach()
        decoder_outputs = torch.stack(tensors=outputs, dim=1)
        return decoder_outputs, hidden

    # input_token.shape 是（batch_size,seq_len)
    # hidden.shape 是(direction*num, batch_size, hidden)
    def forward_step(self, input_token, hidden):
        embedded = self.embedding(input_token)
        # output.shape 是 (batch_size,seq_len,direction*hidden_size)
        # hidden不受 batch_first=True的控制，其shape仍然为(direction*num,batch_size,hidden_size)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.out(hidden[-1]) # hidden[-1]表示最后一个num
        pred_token = logits.argmax(dim = -1,keepdim= True)
        return logits, hidden, pred_token
