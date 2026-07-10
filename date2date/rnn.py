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

    def forward(self, input):
        # 要限定 input的shape
        # input.shape (batch_size, seqlen)
        input = self.embedding(input)
        input = self.dropout(input)
        # hidden不受 batch_first=True的控制，其shape仍然为(direction*num,batch_size,hidden_size)
        # output.shape 是 (batch_size,seq_len,direction*hidden_size)
        output, hidden = self.rnn(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output: Tensor, target_tensor: Optional[Tensor] = None, max_len: int = 20):
        # encoder_hidden 是encoder输出的 hiddeng向量
        # target_tensor.shape is (batch_size, seq_len)
        # foward.shape is (batch_size, seq_len,vocab_size)
        batch_size = encoder_output.shape[0]
        hidden = encoder_output[:, -1: , :].permute(1, 0, 2)
        outputs = []
        # input_token的格式是 (batch_size,seq_len)
        input_token = torch.full(
            (batch_size, 1), SOS_token, dtype=torch.long, device=encoder_output.device
        )  # 设置起始词元
        loop_len = max_len
        if target_tensor is not None:
            loop_len = target_tensor.shape[1]

        for i in range(loop_len):
            logits, hidden, pred_token = self.forward_step(hidden, input_token)
            outputs.append(logits)

            if target_tensor is not None:
                input_token = target_tensor[:,i].unsqueeze(1)
            else :
                input_token = pred_token.detach()
        decoder_outputs = torch.stack(tensors=outputs, dim=1)
        return decoder_outputs, hidden

    def forward_step(self, hidden, input_token):
        # input_token.shape是（batch_size,seq_len)
        embedded = self.embedding(input_token)
        # rnn的hidden输入是(batch_size,seq_len,hidden)
        output, hidden = self.rnn(embedded, hidden)
        # hidden不受 batch_first=True的控制，其shape仍然为(direction*num,batch_size,hidden_size)
        # output.shape 是 (batch_size,seq_len,direction*hidden_size)
        logits = self.out(hidden[-1]) # hidden[-1]表示最后一个num
        pred_token = logits.argmax(dim = -1,keepdim= True)
        return logits, hidden, pred_token
